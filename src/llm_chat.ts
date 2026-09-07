import * as tvmjs from "@mlc-ai/web-runtime";
import * as xgr from "@mlc-ai/web-xgrammar";
import log from "loglevel";
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import { ChatConfig, GenerationConfig, Role } from "./config";
import { getConversation, Conversation } from "./conversation";
import { LogitProcessor, LatencyBreakdown } from "./types";
import {
  getChunkedPrefillInputData,
  getImageDataFromURL,
  getRGBArrayFromImageData,
  getTokenTableFromTokenizer,
} from "./support";
import {
  ChatCompletionFinishReason,
  ChatCompletionTokenLogprob,
  DrowseGenerationFinishReason,
  DrowseReplayLogprob,
  DrowseReplayTokenMetadata,
  TopLogprob,
  ResponseFormat,
  ChatCompletionContentPartImage,
} from "./openai_api_protocols/index";
import {
  AttentionSinkSizeError,
  ContextWindowSizeExceededError,
  MinValueError,
  NonNegativeError,
  RangeError,
  WindowSizeConfigurationError,
  WindowSizeSpecificationError,
  MessageOrderError,
  TextCompletionExpectsKVEmptyError,
  CannotFindImageEmbedError,
  GrammarMatcherInitError,
} from "./error";
import {
  DrowseRankOneProgram,
  DrowseStructuredProgram,
  packDrowseCurveParameters,
  packDrowseGeometryHeader,
  packDrowseGeometryPayload,
  drowseGeometryPayloadElements,
  updatePackedDrowseCurveActive,
  DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE,
  DROWSE_STRUCTURED_MAX_AFFINE_GROUPS,
  DROWSE_STRUCTURED_MAX_CURVES,
  DROWSE_STRUCTURED_MAX_INTRINSIC_DIM,
  DROWSE_STRUCTURED_MAX_PROBES,
  DROWSE_STRUCTURED_MAX_RANK,
  DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES,
  DROWSE_STRUCTURED_MAX_WHITENER_RANK,
  DROWSE_STRUCTURED_GEOMETRY_HEADER_STRIDE,
  DROWSE_STRUCTURED_HOOK_FORMAT,
  DrowseCaptureRow,
  DrowsePreparedCaptureRow,
  DrowseRankOneResidualCaptureV1,
  DrowseResidualCapture,
  DrowseRuntimeCapabilities,
  DrowseStructuredHookProfile,
  DrowseSaeDictionary,
  DrowseJlensDictionary,
  DrowseJlensTopTokenReadout,
  DrowseMeasurementBundle,
  DrowseSaeTopFeatureReadout,
  DROWSE_HOOK_ABI,
  DROWSE_READOUT_TOP_K,
  DROWSE_EXACT_READOUT_ABI,
  DROWSE_EXACT_READOUT_ABI_VERSION,
  DROWSE_EXACT_READOUT_MAX_SAE_FEATURES_PER_CHUNK,
  DROWSE_STRUCTURED_PROFILE_ID,
  DROWSE_STRUCTURED_PROFILE_SCHEMA_VERSION,
  decodeDrowseStructuredHookProfile,
  validateDrowseCaptureRows,
  validateDrowseCapturePositions,
  validateDrowseRankOneProgram,
  validateDrowseSaeDictionary,
  validateDrowseJlensDictionary,
  validateDrowseStructuredProgram,
} from "./drowse";

type ImageURL = ChatCompletionContentPartImage.ImageURL;

type KVStateKind = "kv_cache" | "rnn_state" | "hybrid" | "none";
type ComputeABIKind = "single" | "batch";

type VMFunctionAvailability = {
  prefill: boolean;
  batch_prefill: boolean;
  decode: boolean;
  batch_decode: boolean;
  create_tir_paged_kv_cache: boolean;
  create_rnn_state: boolean;
};

type VMFunctionRegistry = Record<string, tvmjs.PackedFunc | undefined>;

export type DrowseJlensBufferLimits = {
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
};

export type DrowseJlensChunkPlan = {
  layerIds: number[];
  byteLength: number;
};

export const DROWSE_JLENS_MAX_CHUNK_BYTES = 32 * 1024 * 1024;
const DROWSE_JLENS_MAX_MATRICES_PER_CHUNK = 1;
export const DROWSE_SAE_MAX_CHUNK_BYTES = 32 * 1024 * 1024;
export const DROWSE_SAE_MAX_FEATURES_PER_CHUNK =
  DROWSE_EXACT_READOUT_MAX_SAE_FEATURES_PER_CHUNK;

export type DrowseSaeChunkPlan = {
  featureOffset: number;
  featureCount: number;
  paddedFeatureCount: number;
  encoderByteLength: number;
};

export function planDrowseSaeChunks(
  hiddenSize: number,
  featureCount: number,
  limits: DrowseJlensBufferLimits,
): DrowseSaeChunkPlan[] {
  if (!Number.isSafeInteger(hiddenSize) || hiddenSize <= 0) {
    throw new TypeError("Drowse SAE hidden size is invalid");
  }
  if (
    !Number.isSafeInteger(featureCount) ||
    featureCount <= 0 ||
    featureCount > 0x7fffffff
  ) {
    throw new TypeError("Drowse SAE feature count is invalid");
  }
  if (
    !Number.isSafeInteger(limits.maxBufferSize) ||
    limits.maxBufferSize <= 0 ||
    !Number.isSafeInteger(limits.maxStorageBufferBindingSize) ||
    limits.maxStorageBufferBindingSize <= 0
  ) {
    throw new TypeError("Drowse SAE WebGPU buffer limits are invalid");
  }
  const chunkLimit = Math.min(
    limits.maxBufferSize,
    limits.maxStorageBufferBindingSize,
    DROWSE_SAE_MAX_CHUNK_BYTES,
  );
  const featureBytes = hiddenSize * Float32Array.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(featureBytes)) {
    throw new TypeError("Drowse SAE encoder shape is invalid");
  }
  const featuresPerChunk = Math.min(
    Math.floor(chunkLimit / featureBytes),
    DROWSE_SAE_MAX_FEATURES_PER_CHUNK,
  );
  if (featuresPerChunk < DROWSE_READOUT_TOP_K) {
    throw new Error(
      `A Drowse SAE readout chunk requires at least ${DROWSE_READOUT_TOP_K * featureBytes} bytes, exceeding the WebGPU buffer limit ${chunkLimit}`,
    );
  }
  const chunks: DrowseSaeChunkPlan[] = [];
  for (let offset = 0; offset < featureCount; offset += featuresPerChunk) {
    const selected = Math.min(featuresPerChunk, featureCount - offset);
    const padded = Math.max(selected, DROWSE_READOUT_TOP_K);
    chunks.push({
      featureOffset: offset,
      featureCount: selected,
      paddedFeatureCount: padded,
      encoderByteLength: hiddenSize * padded * Float32Array.BYTES_PER_ELEMENT,
    });
  }
  return chunks;
}

export function planDrowseJlensChunks(
  layerIds: readonly number[],
  hiddenSize: number,
  limits: DrowseJlensBufferLimits,
): DrowseJlensChunkPlan[] {
  if (!Number.isSafeInteger(hiddenSize) || hiddenSize <= 0) {
    throw new TypeError("Drowse J-lens hidden size is invalid");
  }
  if (
    !Number.isSafeInteger(limits.maxBufferSize) ||
    limits.maxBufferSize <= 0 ||
    !Number.isSafeInteger(limits.maxStorageBufferBindingSize) ||
    limits.maxStorageBufferBindingSize <= 0
  ) {
    throw new TypeError("Drowse J-lens WebGPU buffer limits are invalid");
  }
  if (
    layerIds.length === 0 ||
    layerIds.some(
      (layerId, index) =>
        !Number.isSafeInteger(layerId) ||
        layerId < 0 ||
        (index > 0 && layerId <= layerIds[index - 1]),
    )
  ) {
    throw new TypeError("Drowse J-lens layer IDs are invalid");
  }
  const matrixElements = hiddenSize * hiddenSize;
  const matrixBytes = matrixElements * Float32Array.BYTES_PER_ELEMENT;
  if (
    !Number.isSafeInteger(matrixElements) ||
    !Number.isSafeInteger(matrixBytes)
  ) {
    throw new TypeError("Drowse J-lens matrix shape is invalid");
  }
  const chunkLimit = Math.min(
    limits.maxBufferSize,
    limits.maxStorageBufferBindingSize,
    DROWSE_JLENS_MAX_CHUNK_BYTES,
  );
  const hardwareLayersPerChunk = Math.floor(chunkLimit / matrixBytes);
  if (hardwareLayersPerChunk < 1) {
    throw new Error(
      `A Drowse J-lens matrix requires ${matrixBytes} bytes, exceeding the WebGPU buffer limit ${chunkLimit}`,
    );
  }
  const layersPerChunk = Math.min(
    hardwareLayersPerChunk,
    DROWSE_JLENS_MAX_MATRICES_PER_CHUNK,
  );
  const chunks: DrowseJlensChunkPlan[] = [];
  for (let start = 0; start < layerIds.length; start += layersPerChunk) {
    const selected = Array.from(layerIds.slice(start, start + layersPerChunk));
    chunks.push({
      layerIds: selected,
      byteLength: selected.length * matrixBytes,
    });
  }
  return chunks;
}

type DrowseRankOneBuffers = {
  enabled: tvmjs.Tensor;
  basis: tvmjs.Tensor;
  neutral: tvmjs.Tensor;
  target: tvmjs.Tensor;
  along: tvmjs.Tensor;
  collapse: tvmjs.Tensor;
  probeBasis: tvmjs.Tensor;
  probeNeutral: tvmjs.Tensor;
};

type DrowseStructuredBuffers = {
  affineActive: tvmjs.Tensor;
  affineBasis: tvmjs.Tensor;
  affineNeutral: tvmjs.Tensor;
  affineTarget: tvmjs.Tensor;
  affineAlong: tvmjs.Tensor;
  affineKappa: tvmjs.Tensor;
  probeKind: tvmjs.Tensor;
  probeDirection: tvmjs.Tensor;
  probeBias: tvmjs.Tensor;
  probeThreshold: tvmjs.Tensor;
  curveBasis?: tvmjs.Tensor;
  curveNeutral?: tvmjs.Tensor;
  curveParameters?: tvmjs.Tensor;
  curveFeet?: tvmjs.Tensor;
  curveDomainKind?: tvmjs.Tensor;
  jLensTokenIds?: tvmjs.Tensor;
  whitenerRank?: tvmjs.Tensor;
  whitenerRidge?: tvmjs.Tensor;
  whitenerBasis?: tvmjs.Tensor;
  whitenerCorrection?: tvmjs.Tensor;
  geometryHeader?: tvmjs.Tensor;
  geometryPayload?: tvmjs.Tensor;
  geometryFeet?: tvmjs.Tensor;
};

type DrowseJlensChunk = {
  layerIds: Int32Array;
  jacobians: tvmjs.Tensor;
  layerIdsDevice: tvmjs.Tensor;
};

type DrowseJlensBuffers = {
  bindingId: string;
  chunks: DrowseJlensChunk[];
};

type DrowseSaeChunk = DrowseSaeChunkPlan & {
  encoder: tvmjs.Tensor;
  encoderBias: tvmjs.Tensor;
  encoderThreshold: tvmjs.Tensor;
  featureOffsetDevice: tvmjs.Tensor;
};

type DrowseSaeBuffers = {
  bindingId: string;
  activation: "relu" | "jump_relu";
  runtimeLayerIndex: number;
  featureCount: number;
  chunks: DrowseSaeChunk[];
  decoderBias: tvmjs.Tensor;
  runtimeLayerIndexDevice: tvmjs.Tensor;
};

type DrowseJlensReadbackChunk = {
  chunk: DrowseJlensChunk;
  selectedHost: tvmjs.Tensor;
  layerTokenIdsHost?: tvmjs.Tensor;
  layerProbabilitiesHost?: tvmjs.Tensor;
};

type DrowsePendingJlensReadback = {
  layerCount: number;
  chunks: DrowseJlensReadbackChunk[];
  aggregate?: {
    tokenIdsHost: tvmjs.Tensor;
    statsHost: tvmjs.Tensor;
    fittedLayerCount: number;
  };
};

type DrowsePendingSaeReadback = {
  valuesHost: tvmjs.Tensor;
  featureIdsHost: tvmjs.Tensor;
  runtimeLayerIndex: number;
  featureCount: number;
};

function cloneDrowseMeasurementBundle(
  value: DrowseMeasurementBundle,
): DrowseMeasurementBundle {
  const jlens = value.jlensTopTokens;
  const sae = value.saeTopFeatures;
  return {
    ...(value.scalar === undefined
      ? {}
      : { scalar: new Float32Array(value.scalar) }),
    ...(value.geometry === undefined
      ? {}
      : { geometry: new Float32Array(value.geometry) }),
    ...(jlens === undefined
      ? {}
      : {
          jlensTopTokens: {
            tokenIds: new Int32Array(jlens.tokenIds),
            strength: new Float32Array(jlens.strength),
            centerOfMass: new Float32Array(jlens.centerOfMass),
            spread: new Float32Array(jlens.spread),
            fittedLayerCount: jlens.fittedLayerCount,
            layerIndices: new Int32Array(jlens.layerIndices),
            layerTokenIds: new Int32Array(jlens.layerTokenIds),
            layerProbabilities: new Float32Array(jlens.layerProbabilities),
          },
        }),
    ...(sae === undefined
      ? {}
      : {
          saeTopFeatures: {
            featureIds: new Int32Array(sae.featureIds),
            activations: new Float32Array(sae.activations),
            runtimeLayerIndex: sae.runtimeLayerIndex,
            featureCount: sae.featureCount,
          },
        }),
  };
}

export interface DrowseSamplerResult {
  sampledTokenId: number;
  sampledLogprob: number;
  selectedLogprobs: DrowseReplayLogprob[];
  argmax: DrowseReplayLogprob;
  topLogprobs: DrowseReplayLogprob[];
  entropyNats: number;
  perplexity: number;
}

export const DROWSE_DEFAULT_TOP_K = 1024;

export function effectiveDrowseTopK(topK: number, vocabSize: number): number {
  if (!Number.isSafeInteger(topK) || topK < 0) {
    throw new Error("Drowse sampler received an invalid top-k value");
  }
  if (!Number.isSafeInteger(vocabSize) || vocabSize < 1) {
    throw new Error("Drowse sampler received an invalid vocabulary size");
  }
  return Math.min(topK > 0 ? topK : DROWSE_DEFAULT_TOP_K, vocabSize);
}

export function sampleDrowseTopKTopP(
  sortedProbabilities: Float32Array,
  sortedTokenIds: Int32Array,
  topP: number,
  topK: number,
  uniformSample: number,
  selectedTokenIds: readonly number[] = [],
): DrowseSamplerResult {
  if (
    sortedProbabilities.length === 0 ||
    sortedProbabilities.length !== sortedTokenIds.length
  ) {
    throw new Error("Drowse sampler received an invalid sorted distribution");
  }
  if (!Number.isFinite(topP) || topP < 0 || topP > 1) {
    throw new Error("Drowse sampler received an invalid top-p value");
  }
  if (!Number.isSafeInteger(topK) || topK < 0) {
    throw new Error("Drowse sampler received an invalid top-k value");
  }
  if (
    !Number.isFinite(uniformSample) ||
    uniformSample < 0 ||
    uniformSample >= 1
  ) {
    throw new Error("Drowse sampler received an invalid uniform draw");
  }

  const candidateCount = effectiveDrowseTopK(topK, sortedProbabilities.length);
  let topKMass = 0;
  for (let index = 0; index < candidateCount; index += 1) {
    const probability = sortedProbabilities[index];
    if (!Number.isFinite(probability) || probability < 0) {
      throw new Error("Drowse sampler received an invalid probability");
    }
    topKMass += probability;
  }
  if (!(topKMass > 0) || !Number.isFinite(topKMass)) {
    throw new Error("Drowse sampler distribution has no probability mass");
  }

  const nucleusTarget = topP * topKMass;
  let supportMass = 0;
  let supportCount = 0;
  for (let index = 0; index < candidateCount; index += 1) {
    supportMass += sortedProbabilities[index];
    supportCount = index + 1;
    if (topP === 0 || supportMass >= nucleusTarget) break;
  }
  if (!(supportMass > 0) || supportCount === 0) {
    throw new Error("Drowse sampler nucleus has no probability mass");
  }

  const logprobAt = (index: number): number => {
    const probability = sortedProbabilities[index];
    return probability === 0
      ? Number.NEGATIVE_INFINITY
      : Math.log(probability / supportMass);
  };
  const requested = new Set(selectedTokenIds);
  const selectedById = new Map<number, number>();
  let sampledTokenId = sortedTokenIds[supportCount - 1];
  let sampledLogprob = logprobAt(supportCount - 1);
  let cumulative = 0;
  let entropyNats = 0;
  for (let index = 0; index < supportCount; index += 1) {
    const tokenId = sortedTokenIds[index];
    const probability = sortedProbabilities[index];
    const logprob = logprobAt(index);
    if (requested.has(tokenId)) selectedById.set(tokenId, logprob);
    const normalized = probability / supportMass;
    if (normalized > 0) entropyNats -= normalized * logprob;
  }
  for (let index = 0; index < supportCount; index += 1) {
    const tokenId = sortedTokenIds[index];
    const probability = sortedProbabilities[index];
    const logprob = logprobAt(index);
    cumulative += probability;
    if (uniformSample < cumulative / supportMass) {
      sampledTokenId = tokenId;
      sampledLogprob = logprob;
      break;
    }
  }
  const selectedLogprobs = selectedTokenIds.map((tokenId) => ({
    token_id: tokenId,
    logprob: selectedById.get(tokenId) ?? Number.NEGATIVE_INFINITY,
  }));
  const topLogprobs = Array.from(
    { length: Math.min(32, supportCount) },
    (_, index) => ({
      token_id: sortedTokenIds[index],
      logprob: logprobAt(index),
    }),
  );
  return {
    sampledTokenId,
    sampledLogprob,
    selectedLogprobs,
    argmax: topLogprobs[0],
    topLogprobs,
    entropyNats,
    perplexity: Math.exp(entropyNats),
  };
}

function samplerLogprobForToken(
  result: DrowseSamplerResult,
  tokenId: number,
): number {
  if (tokenId === result.sampledTokenId) return result.sampledLogprob;
  return (
    result.selectedLogprobs.find((row) => row.token_id === tokenId)?.logprob ??
    result.topLogprobs.find((row) => row.token_id === tokenId)?.logprob ??
    Number.NEGATIVE_INFINITY
  );
}

type ResolvedModelABI = {
  kvStateKind: Exclude<KVStateKind, "none">;
  prefillABI: ComputeABIKind;
  decodeABI: ComputeABIKind;
  prefillFunctionName: "prefill" | "batch_prefill";
  decodeFunctionName: "decode" | "batch_decode";
  needsKVCache: boolean;
  needsRNNState: boolean;
};

export class LLMChatPipeline {
  private config: ChatConfig;
  private tokenizer: Tokenizer;

  // TVM functions
  private tvm: tvmjs.Instance;
  private device: tvmjs.DLDevice;
  private vm: tvmjs.VirtualMachine;
  private prefill: tvmjs.PackedFunc;
  private decoding: tvmjs.PackedFunc;
  private drowsePrefill?: tvmjs.PackedFunc;
  private drowseDecoding?: tvmjs.PackedFunc;
  private drowseStructuredPrefill?: tvmjs.PackedFunc;
  private drowseStructuredDecoding?: tvmjs.PackedFunc;
  private drowseCurvedPrefill?: tvmjs.PackedFunc;
  private drowseCurvedDecoding?: tvmjs.PackedFunc;
  private drowseGeometryPrefill?: tvmjs.PackedFunc;
  private drowseGeometryDecoding?: tvmjs.PackedFunc;
  private drowseCapturePrefill?: tvmjs.PackedFunc;
  private drowseCaptureDecoding?: tvmjs.PackedFunc;
  private drowseRankOneCapturePrefillV1?: tvmjs.PackedFunc;
  private drowseRankOneCaptureDecodingV1?: tvmjs.PackedFunc;
  private drowseJlensReadout?: tvmjs.PackedFunc;
  private drowseJlensReadoutAccumulate?: tvmjs.PackedFunc;
  private drowseJlensReadoutTopK?: tvmjs.PackedFunc;
  private drowseJlensDirections?: tvmjs.PackedFunc;
  private drowseSaeReadoutAccumulate?: tvmjs.PackedFunc;
  private drowseSaeJumpReluReadoutAccumulate?: tvmjs.PackedFunc;
  private drowseHookProfile?: tvmjs.PackedFunc;
  private resolvedModelABI!: ResolvedModelABI;
  private kvStateKind: KVStateKind = "kv_cache";
  private image_embed: tvmjs.PackedFunc | undefined;
  private embed: tvmjs.PackedFunc;
  private fapplyBitmask: tvmjs.PackedFunc;
  private fapplyPenalty: tvmjs.PackedFunc;
  private fapplyLogitBias: tvmjs.PackedFunc;
  private fsoftmaxWithTemperature: tvmjs.PackedFunc;
  private fsampleWithTopP: tvmjs.PackedFunc;
  private fargsortProbs: tvmjs.PackedFunc;

  // Functions related to PagedKVCache
  private fclearKVCaches: tvmjs.PackedFunc;
  private fKVCacheAddSequence: tvmjs.PackedFunc;
  private fKVCacheRemoveSequence: tvmjs.PackedFunc;
  private fKVCacheBeginForward: tvmjs.PackedFunc;
  private fKVCacheEndForward: tvmjs.PackedFunc;
  private fKVCacheEnableSlidingWindowForSeq: tvmjs.PackedFunc;

  // parameter states
  private params: tvmjs.TVMObject;
  private kvCache?: tvmjs.TVMObject = undefined;
  private rnnState?: tvmjs.TVMObject = undefined;
  private prefillLogitPositions?: tvmjs.Tensor = undefined;
  private prefillLogitPositionHost = new Int32Array(1);
  private maxHistorySize = 1;
  private logitsOnCPU?: tvmjs.Tensor = undefined;
  private drowseProgram?: DrowseRankOneBuffers;
  private drowseStructuredProgram?: DrowseStructuredBuffers;
  private drowseStructuredMode?: "structured" | "curved";
  private drowseAffineAlongHost?: Float32Array;
  private drowseCurveParametersHost?: Float32Array;
  private drowseMeasurements?: tvmjs.Tensor;
  private drowseGeometryMeasurements?: tvmjs.Tensor;
  private drowseJlensChunks?: DrowseJlensChunk[];
  private drowseJlensDictionary?: DrowseJlensBuffers;
  private drowseJlensBufferLimits?: DrowseJlensBufferLimits;
  private drowseJlensProbabilitiesHost?: Float32Array;
  private drowseJlensTopTokensHost?: DrowseJlensTopTokenReadout;
  private drowsePendingJlensReadback?: DrowsePendingJlensReadback;
  private drowseSaeDictionary?: DrowseSaeBuffers;
  private drowseSaeReadoutActive = false;
  private drowseSaeTopFeaturesHost?: DrowseSaeTopFeatureReadout;
  private drowsePendingSaeReadback?: DrowsePendingSaeReadback;
  private drowseMeasurementBundleHost?: DrowseMeasurementBundle;
  private drowseExactReadoutAttested = false;
  private drowseProbeKindHost?: Uint32Array;
  private filledKVCacheLength = 0;

  // meta data
  private bosTokenId = 1;
  private contextWindowSize = -1;
  private slidingWindowSize = -1;
  private attentionSinkSize = -1;
  private prefillChunkSize = -1;
  private resetStatsPerPrefill = true;
  private stopStr: string[];
  private stopTokens: Array<number>;

  // states
  private outputMessage = "";
  private outputIds: Array<number> = [];
  private outputPrefix = "";
  private drowseRawMessage = "";
  private stopTriggered = false;
  private finishReason: ChatCompletionFinishReason | undefined = undefined;
  private drowseFinishReason: DrowseGenerationFinishReason | undefined =
    undefined;
  // frequency of appeared token ids till now (refresh after PrefillStep); token_id mapped to freq
  private appearedTokensFreq = new Map<number, number>();
  private imageDataCache = new Map<string, ImageData>();
  private conversation: Conversation;
  // The logprob information of all tokens for this current round (cleared upon each prefillStep)
  // Cleared & updated at the exact same spots as `outputMessage`. Only updated when
  // `genConfig.logprobs` is true. Each entry corresponds to a single autoregressive step.
  private tokenLogprobArray: Array<ChatCompletionTokenLogprob> = [];

  // stats, reset at every `resetChat(keepstats=false)`
  private decodingTotalTime = 0;
  private decodingTotalTokens = 0;
  private prefillTotalTime = 0;
  private prefillTotalTokens = 0;
  // same stats as above, but reset at every `prefillStep()`
  private curRoundDecodingTotalTokens = 0;
  private curRoundPrefillTotalTokens = 0;
  private curRoundDecodingTotalTime = 0;
  private curRoundPrefillTotalTime = 0;
  private curRoundSampledTokens = 0;
  private curRoundDrowseCompletionTokens = 0;

  // additional stats, reset at every prefillStep()
  public curRoundLatencyBreakdown: LatencyBreakdown = {
    logitProcessorTime: [],
    logitBiasTime: [],
    penaltyTime: [],
    sampleTime: [],
    totalTime: [],
    grammarBitmaskTime: [],
  };

  // LogitProcessor
  private logitProcessor?: LogitProcessor = undefined;

  // Grammar-related
  // A grammar matcher for this current round if response_format is set. Reinitialized upon
  // each step regardless of whether the chat is multi-round or not.
  private grammarMatcher?: xgr.GrammarMatcher = undefined;
  // Cache key of the current response_format (schema / grammar / structural tag). If undefined,
  // grammarMatcher is simply using JSON mode. We use this field to determine whether we re-initiate
  // a GrammarMatcher or simply reset during each round (i.e. during prefillStep).
  private responseFormatCacheKey?: string = undefined;
  // A string list of tokens ordered by their token id, post-processed. Once initialized, will not
  // be reinitialized since `this.tokenizer` does not change throughout the lifetime of LLMChatPipeline.
  private xgTokenizerInfo?: xgr.TokenizerInfo = undefined;
  // Compiler for grammar. It is persistent since it specializes on xgTokenizerInfo.
  private grammarCompiler?: xgr.GrammarCompiler = undefined;
  // Size of the bitmask for grammar, determined by fullVocabSize
  private bitmaskSize: number;
  // `vocab_size` read from `config.json`. Can be different from the size of the tokenTable for some
  // models due to dummy padded tokens.
  private fullVocabSize: number;
  // Method to post process the token for grammar; either "byte_level" or default "byte_fallback".
  private token_postproc_method: string;
  // Whether to prepend space for grammar
  private prepend_space_in_encode: boolean;
  // stats for grammar-related overhead
  // Time to initialize grammar matcher in seconds
  private curRoundGrammarInitTotalTime = 0;
  // Total time of getting next bitmask and accepting token in seconds
  private curRoundGrammarPerTokenTotalTime = 0;
  // Instance variables for supporting sampling on WebGPU
  private sampleIndices: Int32Array;
  private sampleIndicesDevice: tvmjs.Tensor;
  private topPDevice: tvmjs.Tensor;

  constructor(
    tvm: tvmjs.Instance,
    tokenizer: Tokenizer,
    config: ChatConfig,
    logitProcessor?: LogitProcessor,
    drowseJlensBufferLimits?: DrowseJlensBufferLimits,
  ) {
    // 0. Setting attributes
    this.tvm = tvm;
    this.tokenizer = tokenizer;
    this.config = config;
    this.logitProcessor = logitProcessor;
    this.drowseJlensBufferLimits = drowseJlensBufferLimits;
    this.fullVocabSize = this.config.vocab_size;
    this.bitmaskSize = Math.ceil(this.fullVocabSize / 32);

    this.conversation = getConversation(
      config.conv_template,
      config.conv_config,
    );
    this.stopStr = this.conversation.getStopStr();
    this.stopTokens = this.conversation.getStopTokens();
    if (config.bos_token_id !== undefined) {
      this.bosTokenId = config.bos_token_id;
    }
    // Set token_post_proc_method, currently mlc-chat-config.json are unstable, hence various
    // fallback mechanisms
    if (config.tokenizer_info !== undefined) {
      this.token_postproc_method = config.tokenizer_info.token_postproc_method;
      this.prepend_space_in_encode =
        config.tokenizer_info.prepend_space_in_encode;
    } else if (config.token_table_postproc_method !== undefined) {
      this.token_postproc_method = config.token_table_postproc_method;
      this.prepend_space_in_encode = false;
    } else {
      log.warn(
        "Cannot find `tokenizer_info` or `token_table_postproc_method` in `mlc-chat-config.json`, " +
          "using default token_postproc_method `raw`.\n" +
          "This field is only used for json mode.",
      );
      this.token_postproc_method = "raw";
      this.prepend_space_in_encode = false;
    }
    log.info("token_postproc_method: ", this.token_postproc_method);
    log.info("prepend_space_in_encode: ", this.prepend_space_in_encode);

    this.device = this.tvm.webgpu();

    // 1. Create VM and read model metadata
    tvm.beginScope();
    this.vm = this.tvm.detachFromCurrentScope(
      this.tvm.createVirtualMachine(this.device),
    );

    const vmFunctionRegistry = LLMChatPipeline.loadVMFunctionRegistry(this.vm, [
      "prefill",
      "batch_prefill",
      "decode",
      "batch_decode",
      "drowse_prefill",
      "drowse_batch_prefill",
      "drowse_decode",
      "drowse_batch_decode",
      "drowse_structured_batch_prefill",
      "drowse_structured_batch_decode",
      "drowse_curved_batch_prefill",
      "drowse_curved_batch_decode",
      "drowse_geometry_batch_prefill",
      "drowse_geometry_batch_decode",
      "drowse_jlens_probabilities",
      "drowse_jlens_readout_accumulate",
      "drowse_jlens_readout_topk",
      "drowse_jlens_directions",
      "drowse_sae_readout_accumulate",
      "drowse_sae_jump_relu_readout_accumulate",
      "drowse_hook_profile",
      "drowse_capture_prefill",
      "drowse_capture_batch_prefill",
      "drowse_capture_decode",
      "drowse_capture_batch_decode",
      "drowse_rank_one_capture_prefill_v1",
      "drowse_rank_one_capture_batch_prefill_v1",
      "drowse_rank_one_capture_decode_v1",
      "drowse_rank_one_capture_batch_decode_v1",
      "create_tir_paged_kv_cache",
      "create_rnn_state",
      "sample_with_top_p",
      "argsort_probs",
      "image_embed",
      "embed",
      "apply_bitmask_inplace",
      "apply_penalty_inplace",
      "apply_logit_bias_inplace",
      "softmax_with_temperature",
    ]);
    LLMChatPipeline.applyLegacyDrowseAliases(vmFunctionRegistry);

    const fgetMetadata = this.vm.getFunction("_metadata");
    const ret_value = fgetMetadata();
    const metadataStr = ret_value.toString();
    const metadata = JSON.parse(metadataStr);
    this.kvStateKind = this.parseKVStateKind(metadata.kv_state_kind);

    const vmFunctionAvailability =
      LLMChatPipeline.getVMFunctionAvailability(vmFunctionRegistry);
    const preferBatchDrowseABI =
      vmFunctionRegistry.drowse_structured_batch_prefill !== undefined &&
      vmFunctionRegistry.drowse_structured_batch_decode !== undefined &&
      vmFunctionRegistry.drowse_curved_batch_prefill !== undefined &&
      vmFunctionRegistry.drowse_curved_batch_decode !== undefined;
    this.resolvedModelABI = LLMChatPipeline.resolveModelABI(
      this.kvStateKind,
      vmFunctionAvailability,
      preferBatchDrowseABI,
    );
    const stateKinds: string[] = [];
    if (this.resolvedModelABI.needsKVCache) {
      stateKinds.push("kv_cache");
    }
    if (this.resolvedModelABI.needsRNNState) {
      stateKinds.push("rnn_state");
    }
    log.info(
      "Resolved model ABI:",
      JSON.stringify({
        kv_state_kind: this.kvStateKind,
        prefill: this.resolvedModelABI.prefillFunctionName,
        decode: this.resolvedModelABI.decodeFunctionName,
        states: stateKinds,
      }),
    );

    // 2. Bind VM functions according to the resolved ABI
    this.prefill = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        this.resolvedModelABI.prefillFunctionName,
        vmFunctionRegistry,
      ),
    );
    this.decoding = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        this.resolvedModelABI.decodeFunctionName,
        vmFunctionRegistry,
      ),
    );
    const drowsePrefillName =
      this.resolvedModelABI.prefillABI === "batch"
        ? "drowse_batch_prefill"
        : "drowse_prefill";
    const drowseDecodeName =
      this.resolvedModelABI.decodeABI === "batch"
        ? "drowse_batch_decode"
        : "drowse_decode";
    const drowsePrefill = vmFunctionRegistry[drowsePrefillName];
    const drowseDecoding = vmFunctionRegistry[drowseDecodeName];
    if (drowsePrefill !== undefined && drowseDecoding !== undefined) {
      this.drowsePrefill = this.tvm.detachFromCurrentScope(drowsePrefill);
      this.drowseDecoding = this.tvm.detachFromCurrentScope(drowseDecoding);
    }
    if (
      this.resolvedModelABI.prefillABI === "batch" &&
      this.resolvedModelABI.decodeABI === "batch"
    ) {
      const structuredPrefill =
        vmFunctionRegistry.drowse_structured_batch_prefill;
      const structuredDecode =
        vmFunctionRegistry.drowse_structured_batch_decode;
      if (structuredPrefill !== undefined && structuredDecode !== undefined) {
        this.drowseStructuredPrefill =
          this.tvm.detachFromCurrentScope(structuredPrefill);
        this.drowseStructuredDecoding =
          this.tvm.detachFromCurrentScope(structuredDecode);
      }
      const curvedPrefill = vmFunctionRegistry.drowse_curved_batch_prefill;
      const curvedDecode = vmFunctionRegistry.drowse_curved_batch_decode;
      if (curvedPrefill !== undefined && curvedDecode !== undefined) {
        this.drowseCurvedPrefill =
          this.tvm.detachFromCurrentScope(curvedPrefill);
        this.drowseCurvedDecoding =
          this.tvm.detachFromCurrentScope(curvedDecode);
      }
      const geometryPrefill = vmFunctionRegistry.drowse_geometry_batch_prefill;
      const geometryDecode = vmFunctionRegistry.drowse_geometry_batch_decode;
      if (geometryPrefill !== undefined && geometryDecode !== undefined) {
        this.drowseGeometryPrefill =
          this.tvm.detachFromCurrentScope(geometryPrefill);
        this.drowseGeometryDecoding =
          this.tvm.detachFromCurrentScope(geometryDecode);
      }
    }
    const jlensReadout = vmFunctionRegistry.drowse_jlens_probabilities;
    if (jlensReadout !== undefined) {
      this.drowseJlensReadout = this.tvm.detachFromCurrentScope(jlensReadout);
    }
    const jlensReadoutAccumulate =
      vmFunctionRegistry.drowse_jlens_readout_accumulate;
    if (jlensReadoutAccumulate !== undefined) {
      this.drowseJlensReadoutAccumulate = this.tvm.detachFromCurrentScope(
        jlensReadoutAccumulate,
      );
    }
    const jlensReadoutTopK = vmFunctionRegistry.drowse_jlens_readout_topk;
    if (jlensReadoutTopK !== undefined) {
      this.drowseJlensReadoutTopK =
        this.tvm.detachFromCurrentScope(jlensReadoutTopK);
    }
    const jlensDirections = vmFunctionRegistry.drowse_jlens_directions;
    if (jlensDirections !== undefined) {
      this.drowseJlensDirections =
        this.tvm.detachFromCurrentScope(jlensDirections);
    }
    const saeReadoutAccumulate =
      vmFunctionRegistry.drowse_sae_readout_accumulate;
    if (saeReadoutAccumulate !== undefined) {
      this.drowseSaeReadoutAccumulate =
        this.tvm.detachFromCurrentScope(saeReadoutAccumulate);
    }
    const saeJumpReluReadoutAccumulate =
      vmFunctionRegistry.drowse_sae_jump_relu_readout_accumulate;
    if (saeJumpReluReadoutAccumulate !== undefined) {
      this.drowseSaeJumpReluReadoutAccumulate = this.tvm.detachFromCurrentScope(
        saeJumpReluReadoutAccumulate,
      );
    }
    const hookProfile = vmFunctionRegistry.drowse_hook_profile;
    if (hookProfile !== undefined) {
      this.drowseHookProfile = this.tvm.detachFromCurrentScope(hookProfile);
    }
    const drowseCapturePrefillName =
      this.resolvedModelABI.prefillABI === "batch"
        ? "drowse_capture_batch_prefill"
        : "drowse_capture_prefill";
    const drowseCaptureDecodeName =
      this.resolvedModelABI.decodeABI === "batch"
        ? "drowse_capture_batch_decode"
        : "drowse_capture_decode";
    const drowseCapturePrefill = vmFunctionRegistry[drowseCapturePrefillName];
    const drowseCaptureDecoding = vmFunctionRegistry[drowseCaptureDecodeName];
    if (
      drowseCapturePrefill !== undefined &&
      drowseCaptureDecoding !== undefined
    ) {
      this.drowseCapturePrefill =
        this.tvm.detachFromCurrentScope(drowseCapturePrefill);
      this.drowseCaptureDecoding = this.tvm.detachFromCurrentScope(
        drowseCaptureDecoding,
      );
    }
    const drowseRankOneCapturePrefillName =
      this.resolvedModelABI.prefillABI === "batch"
        ? "drowse_rank_one_capture_batch_prefill_v1"
        : "drowse_rank_one_capture_prefill_v1";
    const drowseRankOneCaptureDecodeName =
      this.resolvedModelABI.decodeABI === "batch"
        ? "drowse_rank_one_capture_batch_decode_v1"
        : "drowse_rank_one_capture_decode_v1";
    const drowseRankOneCapturePrefill =
      vmFunctionRegistry[drowseRankOneCapturePrefillName];
    const drowseRankOneCaptureDecoding =
      vmFunctionRegistry[drowseRankOneCaptureDecodeName];
    if (
      drowseRankOneCapturePrefill !== undefined &&
      drowseRankOneCaptureDecoding !== undefined
    ) {
      this.drowseRankOneCapturePrefillV1 = this.tvm.detachFromCurrentScope(
        drowseRankOneCapturePrefill,
      );
      this.drowseRankOneCaptureDecodingV1 = this.tvm.detachFromCurrentScope(
        drowseRankOneCaptureDecoding,
      );
    }
    if (this.resolvedModelABI.prefillABI === "batch") {
      log.info("Using batch_prefill kernel.");
    }
    if (this.resolvedModelABI.decodeABI === "batch") {
      log.info("Using batch_decode kernel.");
    }

    this.embed = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName("embed", vmFunctionRegistry),
    );
    this.fapplyBitmask = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        "apply_bitmask_inplace",
        vmFunctionRegistry,
      ),
    );
    this.fapplyPenalty = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        "apply_penalty_inplace",
        vmFunctionRegistry,
      ),
    );
    this.fapplyLogitBias = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        "apply_logit_bias_inplace",
        vmFunctionRegistry,
      ),
    );
    this.fsoftmaxWithTemperature = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        "softmax_with_temperature",
        vmFunctionRegistry,
      ),
    );
    this.fsampleWithTopP = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        "sample_with_top_p",
        vmFunctionRegistry,
      ),
    );
    this.fargsortProbs = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        "argsort_probs",
        vmFunctionRegistry,
      ),
    );
    const imageEmbed = vmFunctionRegistry.image_embed;
    if (imageEmbed !== undefined) {
      this.image_embed = this.tvm.detachFromCurrentScope(imageEmbed);
    } else {
      log.info("Cannot find function image_embed.");
    }

    // 3. Load parameters by name
    const paramNames: string[] = [];
    metadata.params.forEach((param: any) => {
      paramNames.push(param.name);
    });
    this.params = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCacheByName(paramNames),
    );

    // 4. Read in compilation configurations from metadata
    this.prefillChunkSize = metadata.prefill_chunk_size;
    log.info("Using prefillChunkSize: ", this.prefillChunkSize);
    if (this.prefillChunkSize <= 0) {
      throw new MinValueError("prefill_chunk_size", 0);
    }

    // 5. Consolidate KVCache settings: context window, sliding window, attention sink
    this.slidingWindowSize = config.sliding_window_size;
    this.contextWindowSize = config.context_window_size;
    this.attentionSinkSize = config.attention_sink_size;
    if (this.contextWindowSize !== -1 && this.slidingWindowSize !== -1) {
      throw new WindowSizeConfigurationError(
        this.contextWindowSize,
        this.slidingWindowSize,
      );
    } else if (this.slidingWindowSize != -1) {
      // Use sliding window and attention sink
      log.info("Using slidingWindowSize: ", this.slidingWindowSize);
      if (this.attentionSinkSize >= 0) {
        log.info("Using attentionSinkSize: ", this.attentionSinkSize);
      } else {
        throw new AttentionSinkSizeError();
      }
    } else if (this.contextWindowSize != -1) {
      // Use default kv cache without sliding window
      log.info("Using contextWindowSize: ", this.contextWindowSize);
    } else {
      throw new WindowSizeSpecificationError();
    }
    if (
      config.max_history_size !== undefined &&
      config.max_history_size !== null
    ) {
      if (config.max_history_size <= 0) {
        throw new MinValueError("max_history_size", 0);
      }
      this.maxHistorySize = config.max_history_size;
    } else if (this.resolvedModelABI.needsRNNState) {
      // Hybrid/recurrent models can over-allocate RNN state if we use context window directly.
      // Keep browser default conservative unless user explicitly overrides max_history_size.
      log.info("max_history_size is not set. Using browser-safe default: 1");
    }

    // 5. Create cache
    // Load cache functions and instantiate KVCache
    this.fclearKVCaches = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_clear"),
    );
    this.fKVCacheAddSequence = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_add_sequence"),
    );
    this.fKVCacheRemoveSequence = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_remove_sequence"),
    );
    this.fKVCacheBeginForward = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_begin_forward"),
    );
    this.fKVCacheEndForward = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_end_forward"),
    );
    this.fKVCacheEnableSlidingWindowForSeq = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc(
        "vm.builtin.attention_kv_cache_enable_sliding_window_for_seq",
      ),
    );

    const defaultPageSize = 16;
    const defaultMaxNumSequence = 1;
    const maxTotalSeqLen =
      this.slidingWindowSize != -1
        ? this.slidingWindowSize
        : this.contextWindowSize;

    if (this.resolvedModelABI.needsKVCache) {
      const createKVCache = LLMChatPipeline.getRequiredVMFunctionByName(
        "create_tir_paged_kv_cache",
        vmFunctionRegistry,
      );
      this.kvCache = this.tvm.detachFromCurrentScope(
        createKVCache(
          this.tvm.makeShapeTuple([defaultMaxNumSequence]), // max_num_sequence
          this.tvm.makeShapeTuple([maxTotalSeqLen]), // max_total_sequence_length
          this.tvm.makeShapeTuple([this.prefillChunkSize]), // prefill_chunk_size
          this.tvm.makeShapeTuple([defaultPageSize]), // page_size, hard coded for now
          this.tvm.makeShapeTuple([this.slidingWindowSize != -1 ? 1 : 0]),
        ),
      );
    }

    if (this.resolvedModelABI.needsRNNState) {
      const createRNNState = LLMChatPipeline.getRequiredVMFunctionByName(
        "create_rnn_state",
        vmFunctionRegistry,
      );
      this.rnnState = this.tvm.detachFromCurrentScope(
        createRNNState(
          this.tvm.makeShapeTuple([defaultMaxNumSequence]),
          this.tvm.makeShapeTuple([this.maxHistorySize]),
        ),
      );
      log.info("Using maxHistorySize for RNN state: ", this.maxHistorySize);
    }

    if (this.resolvedModelABI.prefillABI === "batch") {
      this.prefillLogitPositions = this.tvm.detachFromCurrentScope(
        this.tvm.empty([defaultMaxNumSequence], "int32", this.device),
      );
    }

    this.filledKVCacheLength = 0;
    this.resetChat(); // especially needed for PagedKVCache as we need to call fKVCacheAddSequence

    // Initialize WebGPU sampling related device tensors
    const numSamples = 1;
    const numProbs = 1;

    this.sampleIndices = new Int32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
      this.sampleIndices[i] = i;
    }
    this.sampleIndicesDevice = this.tvm.detachFromCurrentScope(
      this.tvm
        .empty([numSamples], "int32", this.device)
        .copyFrom(this.sampleIndices),
    );

    this.topPDevice = this.tvm.detachFromCurrentScope(
      this.tvm.empty([numProbs], "float32", this.device),
    );

    tvm.endScope();
  }

  dispose() {
    // TODO: Do we need to dispose all PackedFuncs here?
    this.grammarMatcher?.dispose();
    this.params.dispose();
    this.decoding.dispose();
    this.prefill.dispose();
    this.drowseDecoding?.dispose();
    this.drowsePrefill?.dispose();
    this.drowseStructuredDecoding?.dispose();
    this.drowseStructuredPrefill?.dispose();
    this.drowseCurvedDecoding?.dispose();
    this.drowseCurvedPrefill?.dispose();
    this.drowseGeometryDecoding?.dispose();
    this.drowseGeometryPrefill?.dispose();
    this.drowseCaptureDecoding?.dispose();
    this.drowseCapturePrefill?.dispose();
    this.drowseRankOneCaptureDecodingV1?.dispose();
    this.drowseRankOneCapturePrefillV1?.dispose();
    this.drowseJlensReadout?.dispose();
    this.drowseJlensReadoutAccumulate?.dispose();
    this.drowseJlensReadoutTopK?.dispose();
    this.drowseJlensDirections?.dispose();
    this.drowseSaeReadoutAccumulate?.dispose();
    this.drowseSaeJumpReluReadoutAccumulate?.dispose();
    this.drowseHookProfile?.dispose();
    this.disposeDrowseProgram();
    this.disposeDrowseJlensDictionary();
    this.disposeDrowseSaeDictionary();
    this.embed.dispose();
    this.image_embed?.dispose();
    this.prefillLogitPositions?.dispose();
    this.rnnState?.dispose();
    this.kvCache?.dispose();
    this.fsampleWithTopP.dispose();
    this.fargsortProbs.dispose();
    this.sampleIndicesDevice?.dispose();
    this.topPDevice?.dispose();
    this.vm.dispose();
    this.fclearKVCaches.dispose();
    this.logitsOnCPU?.dispose();
    this.tvm.dispose();
    this.tokenizer.dispose();
    this.xgTokenizerInfo?.dispose();
    this.grammarCompiler?.dispose();
  }

  supportsDrowseRankOneHooks(): boolean {
    return (
      this.drowsePrefill !== undefined && this.drowseDecoding !== undefined
    );
  }

  supportsDrowseStructuredHooks(): boolean {
    return (
      this.drowseHookProfile !== undefined &&
      this.drowseStructuredPrefill !== undefined &&
      this.drowseStructuredDecoding !== undefined
    );
  }

  supportsDrowseCurvedHooks(): boolean {
    return (
      this.drowseCurvedPrefill !== undefined &&
      this.drowseCurvedDecoding !== undefined
    );
  }

  supportsDrowseResidualCapture(): boolean {
    return (
      this.drowseCapturePrefill !== undefined &&
      this.drowseCaptureDecoding !== undefined
    );
  }

  supportsDrowseRankOneResidualCaptureV1(): boolean {
    return (
      this.drowseRankOneCapturePrefillV1 !== undefined &&
      this.drowseRankOneCaptureDecodingV1 !== undefined
    );
  }

  getDrowseRuntimeCapabilities(): DrowseRuntimeCapabilities {
    return {
      topK: true,
      forcedReplay: true,
      replayScoring: true,
      tokenizer: true,
      namedRoles: this.conversation.supportsDrowseNamedRoles(),
      userSeatGeneration: this.conversation.supportsDrowseUserSeatGeneration(),
      sceneStitching: this.conversation.supportsDrowseUserSeatGeneration(),
    };
  }

  async getDrowseStructuredHookProfile(): Promise<DrowseStructuredHookProfile> {
    if (this.drowseHookProfile === undefined) {
      throw new Error(
        "The compiled model does not expose a Drowse hook profile",
      );
    }
    this.tvm.beginScope();
    try {
      const profile = this.drowseHookProfile();
      const host = this.tvm.empty(profile.shape, "int32", this.tvm.cpu());
      host.copyFrom(profile);
      await this.device.sync();
      return decodeDrowseStructuredHookProfile(
        new Int32Array(host.toArray()),
        this.requireModelDimension("num_hidden_layers"),
        this.requireModelDimension("hidden_size"),
      );
    } finally {
      this.tvm.endScope();
    }
  }

  tokenizeDrowseText(text: string): number[] {
    if (typeof text !== "string") {
      throw new TypeError("Drowse tokenizer input must be a string");
    }
    return Array.from(this.tokenizer.encode(text));
  }

  decodeDrowseTokens(tokenIds: readonly number[]): string {
    if (
      !Array.isArray(tokenIds) ||
      tokenIds.some(
        (tokenId) =>
          !Number.isSafeInteger(tokenId) ||
          tokenId < 0 ||
          tokenId >= this.fullVocabSize,
      )
    ) {
      throw new TypeError("Drowse tokenizer IDs must be inside the vocabulary");
    }
    return this.tokenizer.decode(Int32Array.from(tokenIds));
  }

  prepareDrowseCaptureRows(
    rows: DrowseCaptureRow[],
    specialTokenIds: number[],
  ): DrowsePreparedCaptureRow[] {
    validateDrowseCaptureRows(rows, specialTokenIds);
    const special = new Set(specialTokenIds);
    return rows.map((row) => {
      const conversation = getConversation(
        this.config.conv_template,
        this.config.conv_config,
      );
      conversation.override_system_message = row.system;
      for (const message of row.messages) {
        if (message.role === "system") {
          conversation.appendDrowseSystemMessage(message.content);
          continue;
        }
        conversation.appendMessage(
          message.role === "user" ? Role.user : Role.assistant,
          message.content,
          message.roleName,
          true,
        );
      }
      const segments = conversation.getPromptArray(this.config);
      if (segments.some((segment) => typeof segment !== "string")) {
        throw new TypeError(
          "Drowse activation capture supports text rows only",
        );
      }
      const inputIds = [...(conversation.config.system_prefix_token_ids ?? [])];
      let position = -1;
      for (let index = 0; index < segments.length; index += 1) {
        const segment = segments[index] as string;
        const encoded = Array.from(this.tokenizer.encode(segment));
        if (index === segments.length - 1) {
          const messageIndex = row.messages.length - 1;
          const separator =
            conversation.config.seps[
              messageIndex % conversation.config.seps.length
            ];
          if (!segment.endsWith(separator)) {
            throw new Error(
              "The Drowse capture renderer could not identify the final turn separator",
            );
          }
          const withoutSeparator = separator
            ? segment.slice(0, -separator.length)
            : segment;
          const contentPrefix = Array.from(
            this.tokenizer.encode(withoutSeparator),
          );
          if (!isTokenPrefix(contentPrefix, encoded)) {
            throw new Error(
              "The tokenizer changed the assistant content boundary when adding the turn separator",
            );
          }
          position = inputIds.length + contentPrefix.length - 1;
        }
        inputIds.push(...encoded);
      }
      if (inputIds.length === 0 || position < 0) {
        throw new Error("The Drowse capture row rendered to no tokens");
      }
      while (position > 0 && special.has(inputIds[position])) position -= 1;
      return { inputIds, position };
    });
  }

  async captureDrowseResiduals(
    inputIds: number[],
    positions: number[],
  ): Promise<DrowseResidualCapture> {
    if (!this.supportsDrowseResidualCapture()) {
      throw new Error(
        "The loaded model library does not expose Drowse residual capture functions",
      );
    }
    validateDrowseCapturePositions(positions, inputIds.length);
    if (
      this.slidingWindowSize === -1 &&
      inputIds.length > this.contextWindowSize
    ) {
      throw new ContextWindowSizeExceededError(
        inputIds.length,
        this.contextWindowSize,
      );
    }

    const layerCount = this.requireModelDimension("num_hidden_layers");
    const hiddenSize = this.requireModelDimension("hidden_size");
    const values = new Float32Array(layerCount * positions.length * hiddenSize);
    this.resetDrowseCaptureState();
    try {
      for (
        let chunkStart = 0;
        chunkStart < inputIds.length;
        chunkStart += this.prefillChunkSize
      ) {
        const chunk = inputIds.slice(
          chunkStart,
          chunkStart + this.prefillChunkSize,
        );
        const selected: Array<{ outputIndex: number; localPosition: number }> =
          [];
        for (
          let outputIndex = 0;
          outputIndex < positions.length;
          outputIndex += 1
        ) {
          const position = positions[outputIndex];
          if (position >= chunkStart && position < chunkStart + chunk.length) {
            selected.push({
              outputIndex,
              localPosition: position - chunkStart,
            });
          }
        }
        if (selected.length === 0) {
          this.tvm.beginScope();
          try {
            await this.embedAndForward([chunk], chunk.length, false);
          } finally {
            this.tvm.endScope();
          }
          continue;
        }
        const chunkValues = await this.captureDrowseTokenChunk(
          chunk,
          selected.map(({ localPosition }) => localPosition),
        );
        for (let layer = 0; layer < layerCount; layer += 1) {
          for (
            let selectedIndex = 0;
            selectedIndex < selected.length;
            selectedIndex += 1
          ) {
            const source =
              (layer * selected.length + selectedIndex) * hiddenSize;
            const destination =
              (layer * positions.length + selected[selectedIndex].outputIndex) *
              hiddenSize;
            values.set(
              chunkValues.subarray(source, source + hiddenSize),
              destination,
            );
          }
        }
      }
    } finally {
      this.resetDrowseCaptureState();
    }
    return {
      layerCount,
      positionCount: positions.length,
      hiddenSize,
      positions: [...positions],
      values,
    };
  }

  async captureDrowseRankOneResidualsV1(
    inputIds: number[],
    positions: number[],
    program: DrowseRankOneProgram,
  ): Promise<DrowseRankOneResidualCaptureV1> {
    if (!this.supportsDrowseRankOneResidualCaptureV1()) {
      throw new Error(
        "The loaded model library does not expose the Drowse rank-one residual capture v1 ABI",
      );
    }
    if (
      this.drowseProgram !== undefined ||
      this.drowseStructuredProgram !== undefined
    ) {
      throw new Error(
        "Drowse rank-one residual capture requires an idle steering program",
      );
    }
    validateDrowseCapturePositions(positions, inputIds.length);
    if (
      this.slidingWindowSize === -1 &&
      inputIds.length > this.contextWindowSize
    ) {
      throw new ContextWindowSizeExceededError(
        inputIds.length,
        this.contextWindowSize,
      );
    }

    const layerCount = this.requireModelDimension("num_hidden_layers");
    const hiddenSize = this.requireModelDimension("hidden_size");
    const values = new Float32Array(layerCount * positions.length * hiddenSize);
    this.setDrowseRankOneProgram(program);
    this.resetDrowseReadbackState();
    this.resetDrowseCaptureState();
    try {
      for (
        let chunkStart = 0;
        chunkStart < inputIds.length;
        chunkStart += this.prefillChunkSize
      ) {
        const chunk = inputIds.slice(
          chunkStart,
          chunkStart + this.prefillChunkSize,
        );
        const selected: Array<{ outputIndex: number; localPosition: number }> =
          [];
        for (
          let outputIndex = 0;
          outputIndex < positions.length;
          outputIndex += 1
        ) {
          const position = positions[outputIndex];
          if (position >= chunkStart && position < chunkStart + chunk.length) {
            selected.push({
              outputIndex,
              localPosition: position - chunkStart,
            });
          }
        }
        if (selected.length === 0) {
          this.tvm.beginScope();
          try {
            await this.embedAndForward([chunk], chunk.length, true);
          } finally {
            this.tvm.endScope();
          }
          continue;
        }
        const chunkValues = await this.captureDrowseRankOneTokenChunkV1(
          chunk,
          selected.map(({ localPosition }) => localPosition),
        );
        for (let layer = 0; layer < layerCount; layer += 1) {
          for (
            let selectedIndex = 0;
            selectedIndex < selected.length;
            selectedIndex += 1
          ) {
            const source =
              (layer * selected.length + selectedIndex) * hiddenSize;
            const destination =
              (layer * positions.length + selected[selectedIndex].outputIndex) *
              hiddenSize;
            values.set(
              chunkValues.subarray(source, source + hiddenSize),
              destination,
            );
          }
        }
      }
      const measurements = await this.readDrowseMeasurements();
      if (measurements === undefined || measurements.length !== layerCount) {
        throw new Error(
          `Drowse rank-one capture returned ${measurements?.length ?? 0} measurements, ` +
            `expected ${layerCount}`,
        );
      }
      return {
        abiVersion: 1,
        layerCount,
        positionCount: positions.length,
        hiddenSize,
        positions: [...positions],
        values,
        measurements,
      };
    } finally {
      this.resetDrowseCaptureState();
      this.clearDrowseRankOneProgram();
    }
  }

  setDrowseRankOneProgram(program: DrowseRankOneProgram): void {
    if (!this.supportsDrowseRankOneHooks()) {
      throw new Error(
        "The loaded model library does not expose Drowse hook functions",
      );
    }
    const hiddenSize = this.requireModelDimension("hidden_size");
    const layerCount = this.requireModelDimension("num_hidden_layers");
    validateDrowseRankOneProgram(program, hiddenSize, layerCount);

    this.tvm.beginScope();
    let next: DrowseRankOneBuffers;
    try {
      next = {
        enabled: this.tvm
          .empty([layerCount], "uint32", this.device)
          .copyFrom(new Int32Array(program.enabled)),
        basis: this.tvm
          .empty([layerCount, hiddenSize], "float32", this.device)
          .copyFrom(program.basis),
        neutral: this.tvm
          .empty([layerCount, hiddenSize], "float32", this.device)
          .copyFrom(program.neutral),
        target: this.tvm
          .empty([layerCount], "float32", this.device)
          .copyFrom(program.target),
        along: this.tvm
          .empty([layerCount], "float32", this.device)
          .copyFrom(program.along),
        collapse: this.tvm
          .empty([layerCount], "float32", this.device)
          .copyFrom(program.collapse),
        probeBasis: this.tvm
          .empty([layerCount, hiddenSize], "float32", this.device)
          .copyFrom(program.probeBasis),
        probeNeutral: this.tvm
          .empty([layerCount, hiddenSize], "float32", this.device)
          .copyFrom(program.probeNeutral),
      };
      for (const tensor of Object.values(next)) {
        this.tvm.detachFromCurrentScope(tensor);
      }
    } finally {
      this.tvm.endScope();
    }
    this.disposeDrowseProgram();
    this.drowseProgram = next;
  }

  async setDrowseStructuredProgram(
    program: DrowseStructuredProgram,
  ): Promise<void> {
    const hiddenSize = this.requireModelDimension("hidden_size");
    const layerCount = this.requireModelDimension("num_hidden_layers");
    validateDrowseStructuredProgram(program, hiddenSize, layerCount);
    if (
      program.jLensTokenIds?.some((tokenId) => tokenId >= this.fullVocabSize)
    ) {
      throw new TypeError(
        `Drowse J-lens token IDs must be below the model vocabulary size ${this.fullVocabSize}`,
      );
    }
    const hasCurve = program.curveRank.some((rank) => rank > 0);
    const hasGeometry = program.format === DROWSE_STRUCTURED_HOOK_FORMAT;
    if (
      hasCurve
        ? !this.supportsDrowseCurvedHooks()
        : !this.supportsDrowseStructuredHooks()
    ) {
      throw new Error(
        hasCurve
          ? "The loaded model library does not expose Drowse curved hook functions"
          : "The loaded model library does not expose Drowse structured hook functions",
      );
    }
    if (
      hasGeometry &&
      (this.drowseGeometryPrefill === undefined ||
        this.drowseGeometryDecoding === undefined)
    ) {
      throw new Error(
        "The loaded model library does not expose Drowse geometry measurement functions",
      );
    }
    let nextJlensChunks: DrowseJlensChunk[] | undefined;
    if (program.jLensBindingId !== undefined) {
      await this.requireDrowseExactReadout("jlens");
      nextJlensChunks = this.selectDrowseJlensChunks(
        program.jLensBindingId,
        program.jLensLayerIndices!,
      );
    }
    if (program.saeBindingId !== undefined) {
      await this.requireDrowseExactReadout("sae");
      if (this.drowseSaeDictionary?.bindingId !== program.saeBindingId) {
        throw new Error(
          "The Drowse SAE readout program does not match the resident dictionary",
        );
      }
    }

    this.tvm.beginScope();
    let next: DrowseStructuredBuffers;
    let nextCurveParametersHost: Float32Array | undefined;
    try {
      const f32 = (shape: number[], values: Float32Array): tvmjs.Tensor =>
        this.tvm.empty(shape, "float32", this.device).copyFrom(values);
      const u32 = (shape: number[], values: Uint32Array): tvmjs.Tensor =>
        this.tvm
          .empty(shape, "uint32", this.device)
          .copyFrom(new Int32Array(values));
      const i32 = (shape: number[], values: Int32Array): tvmjs.Tensor =>
        this.tvm.empty(shape, "int32", this.device).copyFrom(values);
      next = {
        affineActive: f32(
          [layerCount, DROWSE_STRUCTURED_MAX_AFFINE_GROUPS],
          Float32Array.from(program.affineActive),
        ),
        affineBasis: f32(
          [
            layerCount,
            DROWSE_STRUCTURED_MAX_AFFINE_GROUPS,
            DROWSE_STRUCTURED_MAX_RANK,
            hiddenSize,
          ],
          program.affineBasis,
        ),
        affineNeutral: f32(
          [layerCount, DROWSE_STRUCTURED_MAX_AFFINE_GROUPS, hiddenSize],
          program.affineNeutral,
        ),
        affineTarget: f32(
          [
            layerCount,
            DROWSE_STRUCTURED_MAX_AFFINE_GROUPS,
            DROWSE_STRUCTURED_MAX_RANK,
          ],
          program.affineTarget,
        ),
        affineAlong: f32(
          [layerCount, DROWSE_STRUCTURED_MAX_AFFINE_GROUPS],
          Float32Array.from(
            program.affineAlong,
            (value, index) => value * program.affineActive[index],
          ),
        ),
        affineKappa: f32(
          [
            layerCount,
            DROWSE_STRUCTURED_MAX_AFFINE_GROUPS,
            DROWSE_STRUCTURED_MAX_RANK,
          ],
          program.affineKappa,
        ),
        probeKind: u32(
          [layerCount, DROWSE_STRUCTURED_MAX_PROBES],
          program.probeKind,
        ),
        probeDirection: f32(
          [layerCount, DROWSE_STRUCTURED_MAX_PROBES, hiddenSize],
          program.probeDirection,
        ),
        probeBias: f32(
          [layerCount, DROWSE_STRUCTURED_MAX_PROBES],
          program.probeBias,
        ),
        probeThreshold: f32(
          [layerCount, DROWSE_STRUCTURED_MAX_PROBES],
          program.probeThreshold,
        ),
      };
      if (hasCurve) {
        const curveParameters = packDrowseCurveParameters(program);
        Object.assign(next, {
          curveBasis: f32(
            [
              layerCount,
              DROWSE_STRUCTURED_MAX_CURVES,
              DROWSE_STRUCTURED_MAX_RANK,
              hiddenSize,
            ],
            program.curveBasis,
          ),
          curveNeutral: f32(
            [layerCount, DROWSE_STRUCTURED_MAX_CURVES, hiddenSize],
            program.curveNeutral,
          ),
          curveDomainKind: u32(
            [layerCount, DROWSE_STRUCTURED_MAX_CURVES],
            program.curveDomainKind!,
          ),
          curveParameters: f32(
            [
              layerCount,
              DROWSE_STRUCTURED_MAX_CURVES,
              DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE,
            ],
            curveParameters,
          ),
          curveFeet: f32(
            [
              layerCount,
              DROWSE_STRUCTURED_MAX_CURVES,
              DROWSE_STRUCTURED_MAX_INTRINSIC_DIM,
            ],
            program.curveOrigin,
          ),
        });
        nextCurveParametersHost = curveParameters;
      }
      if (hasGeometry) {
        const geometryHeader = packDrowseGeometryHeader(program);
        const geometryPayload = packDrowseGeometryPayload(program);
        Object.assign(next, {
          whitenerRank: u32([layerCount], program.whitenerRank!),
          whitenerRidge: f32([layerCount], program.whitenerRidge!),
          whitenerBasis: f32(
            [layerCount, DROWSE_STRUCTURED_MAX_WHITENER_RANK, hiddenSize],
            program.whitenerBasis!,
          ),
          whitenerCorrection: f32(
            [layerCount, DROWSE_STRUCTURED_MAX_WHITENER_RANK],
            program.whitenerCorrection!,
          ),
          geometryHeader: u32(
            [
              layerCount,
              DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES,
              DROWSE_STRUCTURED_GEOMETRY_HEADER_STRIDE,
            ],
            geometryHeader,
          ),
          geometryPayload: f32(
            [drowseGeometryPayloadElements(layerCount, hiddenSize)],
            geometryPayload,
          ),
          geometryFeet: f32(
            [
              layerCount,
              DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES,
              DROWSE_STRUCTURED_MAX_INTRINSIC_DIM,
            ],
            program.geometryFeet!,
          ),
        });
      }
      if (program.jLensBindingId !== undefined) {
        if (this.drowseJlensReadout === undefined) {
          throw new Error(
            "The loaded model library does not expose the Drowse J-lens probability function",
          );
        }
        Object.assign(next, {
          jLensTokenIds: i32(
            [DROWSE_STRUCTURED_MAX_PROBES],
            program.jLensTokenIds!,
          ),
        });
      }
      for (const tensor of Object.values(next)) {
        this.tvm.detachFromCurrentScope(tensor);
      }
    } finally {
      this.tvm.endScope();
    }
    this.disposeDrowseProgram();
    this.drowseStructuredProgram = next;
    this.drowseJlensChunks = nextJlensChunks;
    this.drowseSaeReadoutActive = program.saeBindingId !== undefined;
    this.drowseStructuredMode = hasCurve ? "curved" : "structured";
    this.drowseAffineAlongHost = new Float32Array(program.affineAlong);
    this.drowseCurveParametersHost = nextCurveParametersHost;
    this.drowseProbeKindHost = new Uint32Array(program.probeKind);
  }

  async updateDrowseStructuredControls(
    affineActive: Uint32Array,
    curveActive?: Uint32Array,
  ): Promise<void> {
    const program = this.drowseStructuredProgram;
    if (program === undefined) {
      throw new Error("No Drowse structured hook program is installed");
    }
    const layerCount = this.requireModelDimension("num_hidden_layers");
    if (
      affineActive.length !==
      layerCount * DROWSE_STRUCTURED_MAX_AFFINE_GROUPS
    ) {
      throw new TypeError("Drowse affine controls have the wrong length");
    }
    program.affineActive.copyFrom(Float32Array.from(affineActive));
    program.affineAlong.copyFrom(
      Float32Array.from(
        this.drowseAffineAlongHost!,
        (value, index) => value * affineActive[index],
      ),
    );
    if (curveActive !== undefined) {
      if (curveActive.length !== layerCount * DROWSE_STRUCTURED_MAX_CURVES) {
        throw new TypeError(
          "Drowse curve controls do not match the installed program",
        );
      }
      if (program.curveParameters === undefined) {
        if (curveActive.some((value) => value !== 0)) {
          throw new TypeError(
            "Drowse curve controls do not match the installed program",
          );
        }
      } else {
        updatePackedDrowseCurveActive(
          this.drowseCurveParametersHost!,
          curveActive,
        );
        program.curveParameters.copyFrom(this.drowseCurveParametersHost!);
      }
    }
    // WebGPU preserves copyFrom uploads before the next forward on this queue.
    this.resetDrowseReadbackState();
  }

  async setDrowseSaeDictionary(dictionary: DrowseSaeDictionary): Promise<void> {
    await this.requireDrowseExactReadout("sae");
    const hiddenSize = this.requireModelDimension("hidden_size");
    const layerCount = this.requireModelDimension("num_hidden_layers");
    validateDrowseSaeDictionary(dictionary, hiddenSize, layerCount);
    if (
      dictionary.activation === "jump_relu" &&
      this.drowseSaeJumpReluReadoutAccumulate === undefined
    ) {
      throw new Error(
        "The loaded model library does not expose exact JumpReLU SAE readout",
      );
    }
    if (this.drowseSaeDictionary?.bindingId === dictionary.bindingId) {
      if (
        this.drowseSaeDictionary.runtimeLayerIndex !==
          dictionary.runtimeLayerIndex ||
        this.drowseSaeDictionary.featureCount !== dictionary.featureCount
      ) {
        throw new Error(
          "The Drowse SAE binding identity has conflicting metadata",
        );
      }
      return;
    }
    const plan = planDrowseSaeChunks(
      hiddenSize,
      dictionary.featureCount,
      this.requireDrowseJlensBufferLimits(),
    );
    const next = await this.uploadDrowseSaeDictionary(
      dictionary,
      plan,
      hiddenSize,
    );
    this.disposeDrowseProgram();
    this.disposeDrowseSaeDictionary();
    this.drowseSaeDictionary = next;
  }

  async clearDrowseSaeDictionary(): Promise<void> {
    await this.requireDrowseExactReadout("sae");
    if (this.drowseSaeReadoutActive) this.disposeDrowseProgram();
    this.disposeDrowseSaeDictionary();
  }

  async setDrowseJlensDictionary(
    dictionary: DrowseJlensDictionary,
  ): Promise<void> {
    await this.requireDrowseExactReadout("jlens");
    const hiddenSize = this.requireModelDimension("hidden_size");
    const layerCount = this.requireModelDimension("num_hidden_layers");
    validateDrowseJlensDictionary(dictionary, hiddenSize, layerCount);
    if (this.drowseJlensDictionary?.bindingId === dictionary.bindingId) {
      const residentLayers = this.drowseJlensDictionary.chunks.flatMap(
        ({ layerIds }) => Array.from(layerIds),
      );
      if (
        residentLayers.length !== dictionary.layerIndices.length ||
        residentLayers.some(
          (layer, index) => layer !== dictionary.layerIndices[index],
        )
      ) {
        throw new Error(
          "The Drowse J-lens binding identity has conflicting metadata",
        );
      }
      return;
    }
    const plan = planDrowseJlensChunks(
      Array.from(dictionary.layerIndices),
      hiddenSize,
      this.requireDrowseJlensBufferLimits(),
    );
    const chunks = await this.uploadDrowseJlensChunks(
      dictionary.matrices,
      dictionary.layerIndices,
      plan,
      hiddenSize,
    );
    this.disposeDrowseProgram();
    this.disposeDrowseJlensDictionary();
    this.drowseJlensDictionary = {
      bindingId: dictionary.bindingId,
      chunks,
    };
  }

  async clearDrowseJlensDictionary(): Promise<void> {
    await this.requireDrowseExactReadout("jlens");
    if (this.drowseJlensChunks !== undefined) this.disposeDrowseProgram();
    this.disposeDrowseJlensDictionary();
  }

  async readDrowseJlensTopTokens(): Promise<
    DrowseJlensTopTokenReadout | undefined
  > {
    await this.requireDrowseExactReadout("jlens");
    return (await this.readDrowseMeasurementBundle()).jlensTopTokens;
  }

  async readDrowseSaeTopFeatures(): Promise<
    DrowseSaeTopFeatureReadout | undefined
  > {
    await this.requireDrowseExactReadout("sae");
    return (await this.readDrowseMeasurementBundle()).saeTopFeatures;
  }

  clearDrowseRankOneProgram(): void {
    this.disposeDrowseProgram();
  }

  async readDrowseMeasurementBundle(): Promise<DrowseMeasurementBundle> {
    if (this.drowseMeasurementBundleHost !== undefined) {
      return cloneDrowseMeasurementBundle(this.drowseMeasurementBundleHost);
    }
    this.tvm.beginScope();
    try {
      const scalarHost =
        this.drowseMeasurements === undefined
          ? undefined
          : this.tvm
              .empty(this.drowseMeasurements.shape, "float32", this.tvm.cpu())
              .copyFrom(this.drowseMeasurements);
      const geometryHost =
        this.drowseGeometryMeasurements === undefined
          ? undefined
          : this.tvm
              .empty(
                this.drowseGeometryMeasurements.shape,
                "float32",
                this.tvm.cpu(),
              )
              .copyFrom(this.drowseGeometryMeasurements);
      if (
        scalarHost !== undefined ||
        geometryHost !== undefined ||
        this.drowsePendingJlensReadback !== undefined ||
        this.drowsePendingSaeReadback !== undefined
      ) {
        await this.device.sync();
      }
      this.materializeDrowseJlensReadback();
      this.materializeDrowseSaeReadback();
      const scalar =
        scalarHost === undefined
          ? undefined
          : new Float32Array(scalarHost.toArray());
      if (
        scalar !== undefined &&
        this.drowseJlensProbabilitiesHost !== undefined &&
        this.drowseProbeKindHost !== undefined
      ) {
        for (let index = 0; index < scalar.length; index += 1) {
          if (this.drowseProbeKindHost[index] === 3) {
            scalar[index] = this.drowseJlensProbabilitiesHost[index];
          }
        }
      }
      this.drowseMeasurementBundleHost = {
        ...(scalar === undefined ? {} : { scalar }),
        ...(geometryHost === undefined
          ? {}
          : { geometry: new Float32Array(geometryHost.toArray()) }),
        ...(this.drowseJlensTopTokensHost === undefined
          ? {}
          : { jlensTopTokens: this.drowseJlensTopTokensHost }),
        ...(this.drowseSaeTopFeaturesHost === undefined
          ? {}
          : { saeTopFeatures: this.drowseSaeTopFeaturesHost }),
      };
      return cloneDrowseMeasurementBundle(this.drowseMeasurementBundleHost);
    } finally {
      this.disposeDrowsePendingReadbacks();
      this.tvm.endScope();
    }
  }

  async readDrowseMeasurements(): Promise<Float32Array | undefined> {
    return (await this.readDrowseMeasurementBundle()).scalar;
  }

  async readDrowseGeometryMeasurements(): Promise<Float32Array | undefined> {
    return (await this.readDrowseMeasurementBundle()).geometry;
  }

  async resolveDrowseJlensTokenDirections(
    bindingId: string,
    layerIndices: readonly number[],
    tokenIds: readonly number[],
  ): Promise<Float32Array> {
    if (this.drowseJlensDirections === undefined) {
      throw new Error(
        "The compiled model does not expose Drowse J-lens token directions",
      );
    }
    const hiddenSize = this.requireModelDimension("hidden_size");
    const requestedLayers = Int32Array.from(layerIndices);
    const chunks = this.selectDrowseJlensChunks(bindingId, requestedLayers);
    if (
      tokenIds.length === 0 ||
      tokenIds.length > DROWSE_STRUCTURED_MAX_PROBES ||
      tokenIds.some(
        (tokenId) =>
          !Number.isSafeInteger(tokenId) ||
          tokenId < 0 ||
          tokenId >= this.fullVocabSize,
      )
    ) {
      throw new TypeError("Drowse J-lens token IDs are invalid");
    }
    const padded = new Int32Array(DROWSE_STRUCTURED_MAX_PROBES);
    padded.fill(tokenIds[0]);
    padded.set(tokenIds);
    const selected = new Float32Array(
      requestedLayers.length * tokenIds.length * hiddenSize,
    );
    const destinationByLayer = new Map(
      Array.from(requestedLayers, (layer, index) => [layer, index]),
    );
    this.tvm.beginScope();
    try {
      const tokenTensor = this.tvm
        .empty([DROWSE_STRUCTURED_MAX_PROBES], "int32", this.device)
        .copyFrom(padded);
      const pending: Array<{
        chunk: DrowseJlensChunk;
        host: tvmjs.Tensor;
      }> = [];
      for (const chunk of chunks) {
        const directions = this.drowseJlensDirections(
          chunk.jacobians,
          tokenTensor,
          this.params,
        );
        this.requireDrowseTensorShape(
          directions,
          [chunk.layerIds.length, DROWSE_STRUCTURED_MAX_PROBES, hiddenSize],
          "J-lens directions",
        );
        const host = this.tvm.empty(
          directions.shape,
          "float32",
          this.tvm.cpu(),
        );
        host.copyFrom(directions);
        pending.push({ chunk, host });
      }
      await this.device.sync();
      for (const { chunk, host } of pending) {
        const all = new Float32Array(host.toArray());
        for (let row = 0; row < chunk.layerIds.length; row += 1) {
          const source = row * DROWSE_STRUCTURED_MAX_PROBES * hiddenSize;
          const destination = destinationByLayer.get(chunk.layerIds[row])!;
          selected.set(
            all.subarray(source, source + tokenIds.length * hiddenSize),
            destination * tokenIds.length * hiddenSize,
          );
        }
      }
    } finally {
      this.tvm.endScope();
    }
    return selected;
  }

  private async computeDrowseJlensProbabilities(
    hiddenStates: tvmjs.Tensor,
    tokenIds: tvmjs.Tensor,
  ): Promise<void> {
    this.disposeDrowsePendingJlensReadback();
    this.drowseJlensProbabilitiesHost = undefined;
    this.drowseJlensTopTokensHost = undefined;
    if (
      this.drowseJlensReadout === undefined ||
      this.drowseProbeKindHost === undefined ||
      this.drowseJlensChunks === undefined
    ) {
      throw new Error("The Drowse J-lens probability runtime is unavailable");
    }
    if (
      this.drowseExactReadoutAttested &&
      this.drowseJlensReadoutAccumulate !== undefined &&
      this.drowseJlensReadoutTopK !== undefined
    ) {
      await this.computeDrowseJlensProbabilitiesAndTopTokens(
        hiddenStates,
        tokenIds,
      );
      return;
    }
    const layerCount = this.requireModelDimension("num_hidden_layers");
    const pending: DrowseJlensReadbackChunk[] = [];
    this.tvm.beginScope();
    try {
      for (const chunk of this.drowseJlensChunks) {
        const probabilities = this.drowseJlensReadout(
          hiddenStates,
          chunk.jacobians,
          tokenIds,
          chunk.layerIdsDevice,
          this.params,
        );
        this.requireDrowseTensorShape(
          probabilities,
          [chunk.layerIds.length, DROWSE_STRUCTURED_MAX_PROBES],
          "J-lens probabilities",
        );
        const host = this.tvm.empty(
          probabilities.shape,
          "float32",
          this.tvm.cpu(),
        );
        host.copyFrom(probabilities);
        pending.push({ chunk, selectedHost: host });
      }
      for (const { selectedHost } of pending) {
        this.tvm.detachFromCurrentScope(selectedHost);
      }
      this.drowsePendingJlensReadback = { layerCount, chunks: pending };
    } finally {
      this.tvm.endScope();
    }
  }

  private async computeDrowseJlensProbabilitiesAndTopTokens(
    hiddenStates: tvmjs.Tensor,
    tokenIds: tvmjs.Tensor,
  ): Promise<void> {
    const layerCount = this.requireModelDimension("num_hidden_layers");
    const pending: DrowseJlensReadbackChunk[] = [];
    const zero = new Float32Array(this.fullVocabSize);
    let stage = "initializing accumulators";
    this.tvm.beginScope();
    try {
      let probabilitySum = this.tvm
        .empty([this.fullVocabSize], "float32", this.device)
        .copyFrom(zero);
      let depthSum = this.tvm
        .empty([this.fullVocabSize], "float32", this.device)
        .copyFrom(zero);
      let depthSquareSum = this.tvm
        .empty([this.fullVocabSize], "float32", this.device)
        .copyFrom(zero);
      let fittedLayerCount = 0;
      for (const [chunkIndex, chunk] of this.drowseJlensChunks!.entries()) {
        stage = `running chunk ${chunkIndex + 1}/${this.drowseJlensChunks!.length}`;
        const accumulated = this.drowseJlensReadoutAccumulate!(
          hiddenStates,
          chunk.jacobians,
          tokenIds,
          chunk.layerIdsDevice,
          probabilitySum,
          depthSum,
          depthSquareSum,
          this.params,
        );
        const selected = accumulated.get(0) as tvmjs.Tensor;
        const layerTokenIds = accumulated.get(1) as tvmjs.Tensor;
        const layerProbabilities = accumulated.get(2) as tvmjs.Tensor;
        probabilitySum = accumulated.get(3) as tvmjs.Tensor;
        depthSum = accumulated.get(4) as tvmjs.Tensor;
        depthSquareSum = accumulated.get(5) as tvmjs.Tensor;
        this.requireDrowseTensorShape(
          selected,
          [chunk.layerIds.length, DROWSE_STRUCTURED_MAX_PROBES],
          "J-lens probabilities",
        );
        this.requireDrowseTensorShape(
          layerTokenIds,
          [chunk.layerIds.length, DROWSE_READOUT_TOP_K],
          "J-lens layer top-token IDs",
        );
        this.requireDrowseTensorShape(
          layerProbabilities,
          [chunk.layerIds.length, DROWSE_READOUT_TOP_K],
          "J-lens layer top-token probabilities",
        );
        this.requireDrowseTensorShape(
          probabilitySum,
          [this.fullVocabSize],
          "J-lens probability accumulator",
        );
        this.requireDrowseTensorShape(
          depthSum,
          [this.fullVocabSize],
          "J-lens depth accumulator",
        );
        this.requireDrowseTensorShape(
          depthSquareSum,
          [this.fullVocabSize],
          "J-lens depth-square accumulator",
        );
        const selectedHost = this.tvm.empty(
          selected.shape,
          "float32",
          this.tvm.cpu(),
        );
        selectedHost.copyFrom(selected);
        const layerTokenIdsHost = this.tvm.empty(
          layerTokenIds.shape,
          "int32",
          this.tvm.cpu(),
        );
        layerTokenIdsHost.copyFrom(layerTokenIds);
        const layerProbabilitiesHost = this.tvm.empty(
          layerProbabilities.shape,
          "float32",
          this.tvm.cpu(),
        );
        layerProbabilitiesHost.copyFrom(layerProbabilities);
        pending.push({
          chunk,
          selectedHost,
          layerTokenIdsHost,
          layerProbabilitiesHost,
        });
        fittedLayerCount += chunk.layerIds.length;
      }
      stage = "finalizing aggregate top tokens";
      const fittedLayerCountDevice = this.tvm
        .empty([1], "int32", this.device)
        .copyFrom(Int32Array.of(fittedLayerCount));
      const finalized = this.drowseJlensReadoutTopK!(
        probabilitySum,
        depthSum,
        depthSquareSum,
        fittedLayerCountDevice,
      );
      const tokenIdsDevice = finalized.get(0) as tvmjs.Tensor;
      const statsDevice = finalized.get(1) as tvmjs.Tensor;
      this.requireDrowseTensorShape(
        tokenIdsDevice,
        [1, DROWSE_READOUT_TOP_K],
        "J-lens top-token IDs",
      );
      this.requireDrowseTensorShape(
        statsDevice,
        [3, DROWSE_READOUT_TOP_K],
        "J-lens top-token statistics",
      );
      const tokenIdsHost = this.tvm.empty(
        tokenIdsDevice.shape,
        "int32",
        this.tvm.cpu(),
      );
      tokenIdsHost.copyFrom(tokenIdsDevice);
      const statsHost = this.tvm.empty(
        statsDevice.shape,
        "float32",
        this.tvm.cpu(),
      );
      statsHost.copyFrom(statsDevice);
      for (const row of pending) {
        this.tvm.detachFromCurrentScope(row.selectedHost);
        this.tvm.detachFromCurrentScope(row.layerTokenIdsHost!);
        this.tvm.detachFromCurrentScope(row.layerProbabilitiesHost!);
      }
      this.tvm.detachFromCurrentScope(tokenIdsHost);
      this.tvm.detachFromCurrentScope(statsHost);
      this.drowsePendingJlensReadback = {
        layerCount,
        chunks: pending,
        aggregate: { tokenIdsHost, statsHost, fittedLayerCount },
      };
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : typeof error === "object" && error !== null && "message" in error
            ? String(error.message)
            : String(error);
      throw new Error(`Drowse exact J-lens failed while ${stage}: ${message}`);
    } finally {
      this.tvm.endScope();
    }
  }

  private async uploadDrowseJlensChunks(
    matrices: readonly Float32Array[],
    layerIndices: Int32Array,
    plan: readonly DrowseJlensChunkPlan[],
    hiddenSize: number,
  ): Promise<DrowseJlensChunk[]> {
    const matrixElements = hiddenSize * hiddenSize;
    if (layerIndices.length < 1 || matrices.length !== layerIndices.length) {
      throw new TypeError("Drowse J-lens Jacobians have the wrong length");
    }
    const plannedLayers = plan.flatMap(({ layerIds }) => layerIds);
    if (
      plannedLayers.length !== layerIndices.length ||
      plannedLayers.some((layer, index) => layer !== layerIndices[index])
    ) {
      throw new TypeError(
        "Drowse J-lens chunk plan does not match its layer indices",
      );
    }
    this.tvm.beginScope();
    try {
      const chunks: DrowseJlensChunk[] = [];
      let sourceIndex = 0;
      for (const { layerIds } of plan) {
        const hostLayerIds = Int32Array.from(layerIds);
        const values =
          layerIds.length === 1
            ? matrices[sourceIndex]
            : new Float32Array(layerIds.length * matrixElements);
        if (layerIds.length > 1) {
          for (let index = 0; index < layerIds.length; index += 1) {
            values.set(matrices[sourceIndex + index], index * matrixElements);
          }
        }
        chunks.push({
          layerIds: hostLayerIds,
          jacobians: this.tvm
            .empty(
              [layerIds.length, hiddenSize, hiddenSize],
              "float32",
              this.device,
            )
            .copyFrom(values),
          layerIdsDevice: this.tvm
            .empty([layerIds.length], "int32", this.device)
            .copyFrom(hostLayerIds),
        });
        sourceIndex += layerIds.length;
      }
      await this.device.sync();
      for (const chunk of chunks) {
        this.tvm.detachFromCurrentScope(chunk.jacobians);
        this.tvm.detachFromCurrentScope(chunk.layerIdsDevice);
      }
      return chunks;
    } finally {
      this.tvm.endScope();
    }
  }

  private async uploadDrowseSaeDictionary(
    dictionary: DrowseSaeDictionary,
    plan: readonly DrowseSaeChunkPlan[],
    hiddenSize: number,
  ): Promise<DrowseSaeBuffers> {
    this.tvm.beginScope();
    try {
      const decoderBias = this.tvm
        .empty([hiddenSize], "float32", this.device)
        .copyFrom(dictionary.decoderBias);
      const runtimeLayerIndexDevice = this.tvm
        .empty([1], "int32", this.device)
        .copyFrom(Int32Array.of(dictionary.runtimeLayerIndex));
      const chunks: DrowseSaeChunk[] = [];
      for (const chunk of plan) {
        const encoder = new Float32Array(hiddenSize * chunk.paddedFeatureCount);
        for (let row = 0; row < hiddenSize; row += 1) {
          const source = row * dictionary.featureCount + chunk.featureOffset;
          const destination = row * chunk.paddedFeatureCount;
          encoder.set(
            dictionary.encoder.subarray(source, source + chunk.featureCount),
            destination,
          );
        }
        const encoderBias = new Float32Array(chunk.paddedFeatureCount);
        encoderBias.set(
          dictionary.encoderBias.subarray(
            chunk.featureOffset,
            chunk.featureOffset + chunk.featureCount,
          ),
        );
        const encoderThreshold = new Float32Array(chunk.paddedFeatureCount);
        encoderThreshold.set(
          dictionary.encoderThreshold.subarray(
            chunk.featureOffset,
            chunk.featureOffset + chunk.featureCount,
          ),
        );
        const resident: DrowseSaeChunk = {
          ...chunk,
          encoder: this.tvm
            .empty(
              [hiddenSize, chunk.paddedFeatureCount],
              "float32",
              this.device,
            )
            .copyFrom(encoder),
          encoderBias: this.tvm
            .empty([chunk.paddedFeatureCount], "float32", this.device)
            .copyFrom(encoderBias),
          encoderThreshold: this.tvm
            .empty([chunk.paddedFeatureCount], "float32", this.device)
            .copyFrom(encoderThreshold),
          featureOffsetDevice: this.tvm
            .empty([1], "int32", this.device)
            .copyFrom(Int32Array.of(chunk.featureOffset)),
        };
        chunks.push(resident);
      }
      await this.device.sync();
      this.tvm.detachFromCurrentScope(decoderBias);
      this.tvm.detachFromCurrentScope(runtimeLayerIndexDevice);
      for (const chunk of chunks) {
        this.tvm.detachFromCurrentScope(chunk.encoder);
        this.tvm.detachFromCurrentScope(chunk.encoderBias);
        this.tvm.detachFromCurrentScope(chunk.encoderThreshold);
        this.tvm.detachFromCurrentScope(chunk.featureOffsetDevice);
      }
      return {
        bindingId: dictionary.bindingId,
        activation: dictionary.activation,
        runtimeLayerIndex: dictionary.runtimeLayerIndex,
        featureCount: dictionary.featureCount,
        chunks,
        decoderBias,
        runtimeLayerIndexDevice,
      };
    } finally {
      this.tvm.endScope();
    }
  }

  private async computeDrowseSaeTopFeatures(
    hiddenStates: tvmjs.Tensor,
  ): Promise<void> {
    this.disposeDrowsePendingSaeReadback();
    this.drowseSaeTopFeaturesHost = undefined;
    const dictionary = this.drowseSaeDictionary;
    const accumulate =
      dictionary?.activation === "jump_relu"
        ? this.drowseSaeJumpReluReadoutAccumulate
        : this.drowseSaeReadoutAccumulate;
    if (
      !this.drowseSaeReadoutActive ||
      dictionary === undefined ||
      accumulate === undefined
    ) {
      return;
    }
    this.tvm.beginScope();
    try {
      const initialValues = new Float32Array(DROWSE_READOUT_TOP_K);
      initialValues.fill(Number.NEGATIVE_INFINITY);
      let values = this.tvm
        .empty([1, DROWSE_READOUT_TOP_K], "float32", this.device)
        .copyFrom(initialValues);
      let featureIds = this.tvm
        .empty([1, DROWSE_READOUT_TOP_K], "int32", this.device)
        .copyFrom(new Int32Array(DROWSE_READOUT_TOP_K));
      for (const chunk of dictionary.chunks) {
        const accumulated =
          dictionary.activation === "jump_relu"
            ? accumulate(
                hiddenStates,
                chunk.encoder,
                chunk.encoderBias,
                chunk.encoderThreshold,
                dictionary.decoderBias,
                dictionary.runtimeLayerIndexDevice,
                chunk.featureOffsetDevice,
                values,
                featureIds,
              )
            : accumulate(
                hiddenStates,
                chunk.encoder,
                chunk.encoderBias,
                dictionary.decoderBias,
                dictionary.runtimeLayerIndexDevice,
                chunk.featureOffsetDevice,
                values,
                featureIds,
              );
        values = accumulated.get(0) as tvmjs.Tensor;
        featureIds = accumulated.get(1) as tvmjs.Tensor;
        this.requireDrowseTensorShape(
          values,
          [1, DROWSE_READOUT_TOP_K],
          "SAE top-feature values",
        );
        this.requireDrowseTensorShape(
          featureIds,
          [1, DROWSE_READOUT_TOP_K],
          "SAE top-feature IDs",
        );
      }
      const valuesHost = this.tvm.empty(
        values.shape,
        "float32",
        this.tvm.cpu(),
      );
      valuesHost.copyFrom(values);
      const featureIdsHost = this.tvm.empty(
        featureIds.shape,
        "int32",
        this.tvm.cpu(),
      );
      featureIdsHost.copyFrom(featureIds);
      this.tvm.detachFromCurrentScope(valuesHost);
      this.tvm.detachFromCurrentScope(featureIdsHost);
      this.drowsePendingSaeReadback = {
        valuesHost,
        featureIdsHost,
        runtimeLayerIndex: dictionary.runtimeLayerIndex,
        featureCount: dictionary.featureCount,
      };
    } finally {
      this.tvm.endScope();
    }
  }

  private materializeDrowseJlensReadback(): void {
    const pending = this.drowsePendingJlensReadback;
    if (pending === undefined) return;
    const probabilities = new Float32Array(
      pending.layerCount * DROWSE_STRUCTURED_MAX_PROBES,
    );
    for (const { chunk, selectedHost } of pending.chunks) {
      const values = new Float32Array(selectedHost.toArray());
      for (let row = 0; row < chunk.layerIds.length; row += 1) {
        const source = row * DROWSE_STRUCTURED_MAX_PROBES;
        probabilities.set(
          values.subarray(source, source + DROWSE_STRUCTURED_MAX_PROBES),
          chunk.layerIds[row] * DROWSE_STRUCTURED_MAX_PROBES,
        );
      }
    }
    this.drowseJlensProbabilitiesHost = probabilities;
    const aggregate = pending.aggregate;
    if (aggregate === undefined) return;
    const layerIndices = new Int32Array(aggregate.fittedLayerCount);
    const layerTokenIds = new Int32Array(
      aggregate.fittedLayerCount * DROWSE_READOUT_TOP_K,
    );
    const layerProbabilities = new Float32Array(
      aggregate.fittedLayerCount * DROWSE_READOUT_TOP_K,
    );
    let fittedLayerOffset = 0;
    for (const row of pending.chunks) {
      if (
        row.layerTokenIdsHost === undefined ||
        row.layerProbabilitiesHost === undefined
      ) {
        throw new Error("The Drowse J-lens aggregate readback is incomplete");
      }
      layerIndices.set(row.chunk.layerIds, fittedLayerOffset);
      layerTokenIds.set(
        new Int32Array(row.layerTokenIdsHost.toArray()),
        fittedLayerOffset * DROWSE_READOUT_TOP_K,
      );
      layerProbabilities.set(
        new Float32Array(row.layerProbabilitiesHost.toArray()),
        fittedLayerOffset * DROWSE_READOUT_TOP_K,
      );
      fittedLayerOffset += row.chunk.layerIds.length;
    }
    const stats = new Float32Array(aggregate.statsHost.toArray());
    this.drowseJlensTopTokensHost = {
      tokenIds: new Int32Array(aggregate.tokenIdsHost.toArray()),
      strength: stats.slice(0, DROWSE_READOUT_TOP_K),
      centerOfMass: stats.slice(DROWSE_READOUT_TOP_K, 2 * DROWSE_READOUT_TOP_K),
      spread: stats.slice(2 * DROWSE_READOUT_TOP_K),
      fittedLayerCount: aggregate.fittedLayerCount,
      layerIndices,
      layerTokenIds,
      layerProbabilities,
    };
  }

  private materializeDrowseSaeReadback(): void {
    const pending = this.drowsePendingSaeReadback;
    if (pending === undefined) return;
    const allValues = new Float32Array(pending.valuesHost.toArray());
    const allFeatureIds = new Int32Array(pending.featureIdsHost.toArray());
    const selectedFeatureIds: number[] = [];
    const selectedValues: number[] = [];
    for (let index = 0; index < allFeatureIds.length; index += 1) {
      if (
        allFeatureIds[index] >= 0 &&
        allFeatureIds[index] < pending.featureCount
      ) {
        selectedFeatureIds.push(allFeatureIds[index]);
        selectedValues.push(allValues[index]);
      }
    }
    this.drowseSaeTopFeaturesHost = {
      featureIds: Int32Array.from(selectedFeatureIds),
      activations: Float32Array.from(selectedValues),
      runtimeLayerIndex: pending.runtimeLayerIndex,
      featureCount: pending.featureCount,
    };
  }

  private disposeDrowsePendingJlensReadback(): void {
    const pending = this.drowsePendingJlensReadback;
    if (pending === undefined) return;
    for (const row of pending.chunks) {
      row.selectedHost.dispose();
      row.layerTokenIdsHost?.dispose();
      row.layerProbabilitiesHost?.dispose();
    }
    pending.aggregate?.tokenIdsHost.dispose();
    pending.aggregate?.statsHost.dispose();
    this.drowsePendingJlensReadback = undefined;
  }

  private disposeDrowsePendingSaeReadback(): void {
    this.drowsePendingSaeReadback?.valuesHost.dispose();
    this.drowsePendingSaeReadback?.featureIdsHost.dispose();
    this.drowsePendingSaeReadback = undefined;
  }

  private disposeDrowsePendingReadbacks(): void {
    this.disposeDrowsePendingJlensReadback();
    this.disposeDrowsePendingSaeReadback();
  }

  private resetDrowseReadbackState(): void {
    this.drowseMeasurements?.dispose();
    this.drowseMeasurements = undefined;
    this.drowseGeometryMeasurements?.dispose();
    this.drowseGeometryMeasurements = undefined;
    this.disposeDrowsePendingReadbacks();
    this.drowseJlensProbabilitiesHost = undefined;
    this.drowseJlensTopTokensHost = undefined;
    this.drowseSaeTopFeaturesHost = undefined;
    this.drowseMeasurementBundleHost = undefined;
  }

  private requireDrowseTensorShape(
    tensor: tvmjs.Tensor,
    expected: readonly number[],
    label: string,
  ): void {
    if (
      tensor.shape.length !== expected.length ||
      expected.some((size, index) => tensor.shape[index] !== size)
    ) {
      throw new Error(`The Drowse ${label} tensor has the wrong shape`);
    }
  }

  private requireDrowseJlensBufferLimits(): DrowseJlensBufferLimits {
    if (this.drowseJlensBufferLimits === undefined) {
      throw new Error("The Drowse J-lens WebGPU buffer limits are unavailable");
    }
    return this.drowseJlensBufferLimits;
  }

  private selectDrowseJlensChunks(
    bindingId: string,
    layerIndices: Int32Array,
  ): DrowseJlensChunk[] {
    const dictionary = this.drowseJlensDictionary;
    if (dictionary === undefined || dictionary.bindingId !== bindingId) {
      throw new Error(
        "The Drowse J-lens program does not match the resident dictionary",
      );
    }
    if (
      layerIndices.length < 1 ||
      layerIndices.some(
        (layer, index) =>
          layer < 0 || (index > 0 && layer <= layerIndices[index - 1]),
      )
    ) {
      throw new TypeError(
        "Drowse J-lens layer indices must be unique and ordered",
      );
    }
    const byLayer = new Map<number, DrowseJlensChunk>();
    for (const chunk of dictionary.chunks) {
      if (chunk.layerIds.length !== 1) {
        throw new Error("The resident Drowse J-lens chunk layout is invalid");
      }
      byLayer.set(chunk.layerIds[0], chunk);
    }
    return Array.from(layerIndices, (layer) => {
      const chunk = byLayer.get(layer);
      if (chunk === undefined) {
        throw new Error(
          `The resident Drowse J-lens has no matrix for runtime layer ${layer}`,
        );
      }
      return chunk;
    });
  }

  private async requireDrowseExactReadout(
    family: "jlens" | "sae",
  ): Promise<void> {
    if (
      family === "jlens" &&
      (this.drowseJlensReadoutAccumulate === undefined ||
        this.drowseJlensReadoutTopK === undefined)
    ) {
      throw new Error(
        "The compiled model does not expose the exact Drowse J-lens readout functions",
      );
    }
    if (family === "sae" && this.drowseSaeReadoutAccumulate === undefined) {
      throw new Error(
        "The compiled model does not expose the exact Drowse SAE readout function",
      );
    }
    if (this.drowseExactReadoutAttested) return;
    const profile = await this.getDrowseStructuredHookProfile();
    if (
      profile.schemaVersion !== DROWSE_STRUCTURED_PROFILE_SCHEMA_VERSION ||
      profile.id !== DROWSE_STRUCTURED_PROFILE_ID ||
      profile.hookAbi !== DROWSE_HOOK_ABI ||
      profile.exactReadoutAbiVersion !== DROWSE_EXACT_READOUT_ABI_VERSION ||
      profile.exactReadoutAbi !== DROWSE_EXACT_READOUT_ABI ||
      profile.readoutTopK !== DROWSE_READOUT_TOP_K ||
      profile.maxSaeFeaturesPerChunk !== DROWSE_SAE_MAX_FEATURES_PER_CHUNK
    ) {
      throw new Error(
        "The compiled model does not attest the exact Drowse readout ABI",
      );
    }
    this.drowseExactReadoutAttested = true;
  }

  private requireModelDimension(field: string): number {
    const modelConfig = this.config.model_config;
    const textConfig = modelConfig?.text_config as
      | Record<string, unknown>
      | undefined;
    const value = modelConfig?.[field] ?? textConfig?.[field];
    if (!Number.isSafeInteger(value) || (value as number) <= 0) {
      throw new Error(`The model config does not contain a valid ${field}`);
    }
    return value as number;
  }

  private disposeDrowseProgram(): void {
    this.resetDrowseReadbackState();
    this.drowseJlensChunks = undefined;
    this.drowseSaeReadoutActive = false;
    this.drowseProbeKindHost = undefined;
    this.drowseAffineAlongHost = undefined;
    this.drowseCurveParametersHost = undefined;
    if (this.drowseProgram !== undefined) {
      for (const tensor of Object.values(this.drowseProgram)) tensor.dispose();
      this.drowseProgram = undefined;
    }
    if (this.drowseStructuredProgram !== undefined) {
      for (const tensor of Object.values(this.drowseStructuredProgram))
        tensor.dispose();
      this.drowseStructuredProgram = undefined;
    }
    this.drowseStructuredMode = undefined;
  }

  private disposeDrowseJlensDictionary(): void {
    const dictionary = this.drowseJlensDictionary;
    if (dictionary !== undefined) {
      for (const chunk of dictionary.chunks) {
        chunk.jacobians.dispose();
        chunk.layerIdsDevice.dispose();
      }
      this.drowseJlensDictionary = undefined;
    }
    this.drowseJlensChunks = undefined;
    this.disposeDrowsePendingJlensReadback();
    this.drowseJlensProbabilitiesHost = undefined;
    this.drowseJlensTopTokensHost = undefined;
    this.drowseMeasurementBundleHost = undefined;
  }

  private disposeDrowseSaeDictionary(): void {
    const dictionary = this.drowseSaeDictionary;
    if (dictionary !== undefined) {
      dictionary.decoderBias.dispose();
      dictionary.runtimeLayerIndexDevice.dispose();
      for (const chunk of dictionary.chunks) {
        chunk.encoder.dispose();
        chunk.encoderBias.dispose();
        chunk.encoderThreshold.dispose();
        chunk.featureOffsetDevice.dispose();
      }
      this.drowseSaeDictionary = undefined;
    }
    this.drowseSaeReadoutActive = false;
    this.disposeDrowsePendingSaeReadback();
    this.drowseSaeTopFeaturesHost = undefined;
    this.drowseMeasurementBundleHost = undefined;
  }

  /**
   * Get the current message.
   */
  getMessage() {
    return this.outputMessage;
  }

  /** Return the monotonically decoded generated tokens before stop-string trimming. */
  getDrowseRawMessage(): string {
    return this.drowseRawMessage;
  }

  /**
   * Reset the runtime statistics
   */
  resetRuntimeStats() {
    this.prefillTotalTime = 0;
    this.prefillTotalTokens = 0;
    this.decodingTotalTime = 0;
    this.decodingTotalTokens = 0;
  }

  /**
   * Reset the chat history
   */
  resetChat(keepStats = false) {
    this.tvm.beginScope();
    this.conversation.reset();
    if (!keepStats) {
      this.resetRuntimeStats();
    }
    this.resetKVCache();
    this.filledKVCacheLength = 0;
    this.outputPrefix = "";
    this.logitProcessor?.resetState();
    this.tvm.endScope();
  }

  /**
   * Reset KV Cache
   */
  resetKVCache() {
    const states = this.getActiveKVStates();
    for (const state of states) {
      this.fclearKVCaches(state);
      this.fKVCacheAddSequence!(state, new tvmjs.Scalar(0, "int64"));
    }
    if (this.slidingWindowSize != -1 && this.kvCache !== undefined) {
      this.fKVCacheEnableSlidingWindowForSeq(
        this.kvCache,
        new tvmjs.Scalar(0, "int64"),
        new tvmjs.Scalar(this.slidingWindowSize, "int32"),
        new tvmjs.Scalar(this.attentionSinkSize, "int32"),
      );
    }
  }

  /**
   * @returns Whether stop is triggered.
   */
  stopped(): boolean {
    return this.stopTriggered;
  }

  /**
   * @returns Finish reason; undefined if generation not started/stopped yet.
   */
  getFinishReason(): ChatCompletionFinishReason | undefined {
    return this.finishReason;
  }

  getDrowseFinishReason(): DrowseGenerationFinishReason | undefined {
    return this.drowseFinishReason;
  }

  /**
   * @returns tokenLogprobArray for this current round of autoregressive generation.
   * Updated upon each sampled token, cleared upon each prefillStep().
   */
  getTokenLogprobArray(): Array<ChatCompletionTokenLogprob> {
    return this.tokenLogprobArray;
  }

  /**
   * @returns the number of tokens decoded for a single request or a single choice in the request.
   */
  getCurRoundDecodingTotalTokens(): number {
    return this.curRoundDecodingTotalTokens;
  }

  getCurRoundDrowseCompletionTokens(): number {
    return this.curRoundDrowseCompletionTokens;
  }

  /**
   * @returns the number of tokens decoded for a single request or a single choice in the request.
   */
  getCurRoundPrefillTotalTokens(): number {
    return this.curRoundPrefillTotalTokens;
  }

  /**
   * @returns the time spent on decode for a single request or a single choice in the request.
   */
  getCurRoundDecodingTotalTime(): number {
    return this.curRoundDecodingTotalTime;
  }

  /**
   * @returns the time spent on  for a single request or a single choice in the request.
   */
  getCurRoundPrefillTotalTime(): number {
    return this.curRoundPrefillTotalTime;
  }

  /**
   * @returns the time (seconds) spent on for initializing grammar matcher for a single request.
   */
  getCurRoundGrammarInitTotalTime(): number {
    return this.curRoundGrammarInitTotalTime;
  }

  /**
   * @returns the total time (seconds) spent on creating bitmask and accepting token grammar matcher
   * for all the generated tokens in a single request.
   */
  getCurRoundGrammarPerTokenTotalTime(): number {
    return this.curRoundGrammarPerTokenTotalTime;
  }

  /**
   * @returns the breakdown of latencies for sampling each token for a single request.
   */
  getCurRoundLatencyBreakdown(): LatencyBreakdown {
    return this.curRoundLatencyBreakdown;
  }

  /**
   * @returns Runtime stats information.
   */
  runtimeStatsText(): string {
    return (
      `prefill: ${(this.prefillTotalTokens / this.prefillTotalTime).toFixed(4)} tokens/sec, ` +
      `decoding: ${(this.decodingTotalTokens / this.decodingTotalTime).toFixed(4)} tokens/sec`
    );
  }

  /**
   * @returns Runtime stats information, starting from the last prefill performed.
   */
  curRoundRuntimeStatsText(): string {
    return (
      `prefill: ${this.getCurRoundPrefillTokensPerSec().toFixed(4)} tokens/sec, ` +
      `decoding: ${this.getCurRoundDecodingTokensPerSec().toFixed(4)} tokens/sec`
    );
  }

  /**
   * @returns Prefill tokens per second, starting from the last prefill performed.
   */
  getCurRoundPrefillTokensPerSec(): number {
    return this.curRoundPrefillTotalTokens / this.curRoundPrefillTotalTime;
  }

  /**
   * @returns Prefill tokens per second, starting from the last prefill performed.
   */
  getCurRoundDecodingTokensPerSec(): number {
    return this.curRoundDecodingTotalTokens / this.curRoundDecodingTotalTime;
  }

  /**
   * Set the seed for the RNG `this.tvm.rng`.
   */
  setSeed(seed: number): void {
    this.tvm.setSeed(seed);
  }

  private getResponseFormatKey(
    responseFormat?: ResponseFormat | null,
  ): string | undefined {
    if (!responseFormat) {
      return undefined;
    }
    if (responseFormat.type === "json_object") {
      return responseFormat.schema ?? undefined;
    }
    if (responseFormat.type === "grammar") {
      return responseFormat.grammar ?? undefined;
    }
    if (responseFormat.type === "structural_tag") {
      const structuralTag = responseFormat.structural_tag;
      if (structuralTag === undefined || structuralTag === null) {
        return undefined;
      }
      return typeof structuralTag === "string"
        ? structuralTag
        : JSON.stringify(structuralTag);
    }
    return undefined;
  }

  // Getters and setters for this.conversation.
  /**
   * @returns The conversation object (not a deep copy).
   */
  getConversationObject(): Conversation {
    return this.conversation;
  }

  /**
   * Set this.conversation to a new conversation object.
   */
  setConversation(newConv: Conversation) {
    this.conversation = newConv;
    this.stopStr = this.conversation.getStopStr();
    this.stopTokens = this.conversation.getStopTokens();
  }

  async asyncLoadWebGPUPipelines() {
    await this.tvm.asyncLoadWebGPUPipelines(this.vm.getInternalModule());
  }

  /**
   * Generate the first token given input prompt
   */
  async prefillStep(
    inp: string,
    msgRole: Role,
    inp_role_str?: string,
    genConfig?: GenerationConfig,
  ): Promise<void> {
    if (
      msgRole !== Role.user &&
      msgRole !== Role.assistant &&
      msgRole !== Role.tool
    ) {
      throw new MessageOrderError(
        "The last message has an unsupported generation seat.",
      );
    }
    if (this.resetStatsPerPrefill) {
      this.resetRuntimeStats();
    }

    const tstart = performance.now();

    // cleanup the per convo states
    this.outputIds = [];
    this.outputPrefix = "";
    this.appearedTokensFreq.clear();
    this.outputMessage = "";
    this.drowseRawMessage = "";
    this.tokenLogprobArray = [];
    this.curRoundDecodingTotalTokens = 0;
    this.curRoundPrefillTotalTokens = 0;
    this.curRoundPrefillTotalTime = 0;
    this.curRoundDecodingTotalTime = 0;
    this.curRoundGrammarInitTotalTime = 0;
    this.curRoundGrammarPerTokenTotalTime = 0;
    this.curRoundSampledTokens = 0;
    this.curRoundDrowseCompletionTokens = 0;

    this.curRoundLatencyBreakdown = {
      logitProcessorTime: [],
      logitBiasTime: [],
      penaltyTime: [],
      sampleTime: [],
      totalTime: [],
      grammarBitmaskTime: [],
    };

    this.stopTriggered = false;
    this.finishReason = undefined;
    this.drowseFinishReason = undefined;
    const conversation = this.conversation;

    // -1. Instantiate grammar matcher according to generation config. This step is overlapped
    // with prefilling the prompt to hide overhead by using this promise.
    let grammarMatcherInitPromise: Promise<void> | undefined = undefined;
    const responseFormat = genConfig?.response_format;
    if (
      responseFormat?.type === "json_object" ||
      responseFormat?.type === "grammar" ||
      responseFormat?.type === "structural_tag"
    ) {
      const curResponseFormatKey = this.getResponseFormatKey(responseFormat);
      if (
        curResponseFormatKey === this.responseFormatCacheKey &&
        this.grammarMatcher
      ) {
        // If we did not change the schema and have instantiated a GrammarMatcher, we reuse it.
        const tGrammarInitStart = performance.now();
        log.info("Reuse grammar matcher.");
        this.grammarMatcher.reset();
        this.curRoundGrammarInitTotalTime =
          (performance.now() - tGrammarInitStart) / 1e3;
      } else {
        // Else dispose current grammarMatcher, reinitialize, and update this.schema.
        grammarMatcherInitPromise = (async () => {
          const tGrammarInitStart = performance.now();
          log.info("Initialize new grammar matcher.");
          try {
            if (this.grammarMatcher) {
              this.grammarMatcher.dispose();
              this.grammarMatcher = undefined;
            }
            if (this.xgTokenizerInfo === undefined) {
              log.info("Initialize token table.");
              // Post process entire table
              const rawTokenTable = getTokenTableFromTokenizer(this.tokenizer);
              this.xgTokenizerInfo =
                await xgr.TokenizerInfo.createTokenizerInfo(
                  rawTokenTable,
                  this.token_postproc_method,
                  this.prepend_space_in_encode,
                  this.fullVocabSize,
                  this.stopTokens,
                );
              this.grammarCompiler =
                await xgr.GrammarCompiler.createGrammarCompiler(
                  this.xgTokenizerInfo,
                );
            }
            const grammar: xgr.CompiledGrammar =
              responseFormat.type === undefined
                ? await this.grammarCompiler!.compileBuiltinJSONGrammar()
                : responseFormat.type === "json_object"
                  ? await this.grammarCompiler!.compileJSONSchema(
                      responseFormat.schema!,
                    )
                  : responseFormat.type === "grammar"
                    ? await this.grammarCompiler!.compileGrammar(
                        responseFormat.grammar!,
                      )
                    : await this.grammarCompiler!.compileStructuralTag(
                        responseFormat.structural_tag!,
                      );
            try {
              this.grammarMatcher =
                await xgr.GrammarMatcher.createGrammarMatcher(grammar);
            } finally {
              grammar.dispose();
            }
            this.responseFormatCacheKey = curResponseFormatKey;
            this.curRoundGrammarInitTotalTime =
              (performance.now() - tGrammarInitStart) / 1e3;
          } catch (err) {
            throw new GrammarMatcherInitError(
              responseFormat.type ?? "json_object",
              err,
            );
          }
        })();
        // Grammar initialization overlaps prompt prefill, so attach a handler now. The original
        // promise still rejects when awaited below, but browsers will not report it as unhandled
        // if initialization fails before prefill finishes.
        void grammarMatcherInitPromise.catch(() => undefined);
      }
    }

    // 0. Get inputData from conversation
    if (conversation.isTextCompletion) {
      conversation.prompt = inp;
    } else {
      conversation.appendMessage(msgRole, inp, inp_role_str);
      const generationSeat =
        genConfig?.drowse_generation_seat === "user"
          ? Role.user
          : Role.assistant;
      if (genConfig?.enable_thinking === false) {
        // TODO(Charlie): In future we should make emptyThinkingBlockStr configurable.
        const emptyThinkingBlockStr = "<think>\n\n</think>\n\n";
        this.outputPrefix = emptyThinkingBlockStr;
        conversation.appendEmptyThinkingReplyHeader(
          generationSeat,
          emptyThinkingBlockStr,
          genConfig.drowse_generation_role ?? undefined,
        );
      } else {
        if (genConfig?.drowse_generation_role === undefined) {
          conversation.appendReplyHeader(generationSeat);
        } else {
          conversation.appendReplyHeader(
            generationSeat,
            genConfig.drowse_generation_role ?? undefined,
          );
        }
      }
    }
    const [inputData, promptLen, getEmbedSize] = await this.getInputData();

    // Check if LLMChatPipeline fits for forwarding image input
    const hasImageInput = inputData.some((data) => !Array.isArray(data));
    if (hasImageInput && this.image_embed === undefined) {
      throw new CannotFindImageEmbedError();
    }

    // 1. Chunk inputData to embed and forward in one shot for each, minimize intermediate data
    const retGetChunks = getChunkedPrefillInputData(
      inputData,
      this.prefillChunkSize,
      getEmbedSize,
    );
    const chunks: Array<Array<number> | ImageURL>[] = retGetChunks[0];
    const chunkLens: Array<number> = retGetChunks[1];

    // 2. Prefill each chunk
    this.tvm.beginScope();
    let logits: tvmjs.Tensor;
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const chunkLen = chunkLens[i];
      const prevFilledLen = this.filledKVCacheLength;
      logits = this.tvm.detachFromCurrentScope(
        await this.embedAndForward(chunk, chunkLen),
      );
      if (this.filledKVCacheLength !== prevFilledLen + chunkLen) {
        throw new Error(
          "Internal Error: filledKVCacheLength does not match expected value.",
        );
      }
    }
    this.imageDataCache.clear();
    this.tvm.endScope();

    // 4. Sample, stats, post process token sampled.
    // Grammar initialization has already been running alongside prefill. Wait for the GPU first
    // so logits can be disposed safely even when grammar initialization fails.
    let nextToken: number;
    try {
      await this.device.sync();
      await grammarMatcherInitPromise;
      nextToken = await this.sampleTokenFromLogits(logits!, genConfig);
    } finally {
      logits!.dispose();
    }
    const tend = performance.now();

    this.prefillTotalTime += (tend - tstart) / 1e3;
    this.prefillTotalTokens += promptLen;
    this.curRoundPrefillTotalTokens += promptLen;
    this.curRoundPrefillTotalTime += (tend - tstart) / 1e3;

    this.processNextToken(nextToken, genConfig);
  }

  async decodeStep(genConfig?: GenerationConfig): Promise<void> {
    if (this.stopTriggered) {
      throw Error("Cannot run decode when stopped");
    }

    const tstart = performance.now();

    this.tvm.beginScope();
    const chunk: Array<Array<number>> = [
      this.outputIds.slice(this.outputIds.length - 1),
    ];
    const chunkLen = chunk.length;
    const prevFilledLen = this.filledKVCacheLength;
    const logits = this.tvm.detachFromCurrentScope(
      await this.embedAndForward(chunk, chunkLen),
    );
    if (this.filledKVCacheLength !== prevFilledLen + chunkLen) {
      throw new Error(
        "Internal Error: filledKVCacheLength does not match expected value.",
      );
    }
    this.tvm.endScope();

    // sample from logits
    const nextToken = await this.sampleTokenFromLogits(logits, genConfig);
    logits.dispose();
    const tend = performance.now();

    this.decodingTotalTime += (tend - tstart) / 1e3;
    this.decodingTotalTokens += 1;
    this.curRoundDecodingTotalTokens += 1;
    this.curRoundDecodingTotalTime += (tend - tstart) / 1e3;

    this.processNextToken(nextToken, genConfig);
  }

  /**
   * Manually trigger stop if it is not stopped.
   */
  triggerStop() {
    if (this.stopTriggered) {
      return;
    }
    this.stopTriggered = true;
    this.finishReason = "abort";
    this.drowseFinishReason = "external_stop";
    if (!this.conversation.isTextCompletion) {
      this.conversation.finishReply(this.outputPrefix + this.outputMessage);
    }
  }

  /**
   * Add a generated token and check for stop.
   *
   * @param nextToken The next token.
   * @param genConfig Configs that override `this.config` for this round of generation.
   */
  private processNextToken(
    nextToken: number,
    genConfig?: GenerationConfig,
  ): void {
    if (this.stopTriggered) {
      throw Error("Cannot call process when it is stoppped");
    }

    // Get max_tokens from generationConfig (specified by user in completion request)
    // If not specified, do not set a limit
    const max_tokens = genConfig?.max_tokens ?? Infinity;
    if (!(max_tokens > 0)) {
      throw new MinValueError("max_tokens", 0);
    }

    // Get ignore_eos from generationConfig (specified by user in completion request)
    let ignore_eos = false;
    if (
      genConfig !== undefined &&
      genConfig.ignore_eos !== undefined &&
      genConfig.ignore_eos !== null
    ) {
      ignore_eos = genConfig.ignore_eos;
    }

    // Get stopStrs, possibly overridden by genConfig for this round
    let stopStrs = this.stopStr;
    if (genConfig !== undefined && genConfig.stop) {
      stopStrs = stopStrs.concat(genConfig.stop);
    }

    let stopTokens = this.stopTokens;
    if (ignore_eos) {
      stopTokens = [];
      stopStrs = [];
    }

    // Stop condition 1: stop token; otherwise, append to `this.outputIds`
    if (stopTokens.includes(nextToken)) {
      this.stopTriggered = true;
      this.finishReason = "stop";
      this.drowseFinishReason = "eos";
    }
    if (!this.stopTriggered) {
      this.outputIds.push(nextToken);
      this.curRoundDrowseCompletionTokens += 1;
      // Update token appearance frequency
      const curFreq = this.appearedTokensFreq.get(nextToken);
      if (curFreq !== undefined) {
        this.appearedTokensFreq.set(nextToken, curFreq + 1);
      } else {
        this.appearedTokensFreq.set(nextToken, 1);
      }
    }

    // Stop condition 2: stop string; update `this.outputMessage` subsequently
    let outputMessage = this.tokenizer.decode(new Int32Array(this.outputIds));
    this.drowseRawMessage = outputMessage;
    let stopPos = -1;
    for (const stopStr of stopStrs) {
      // Stop at the first stopStr we find
      stopPos = outputMessage.lastIndexOf(stopStr);
      if (stopPos != -1) {
        outputMessage = outputMessage.substring(0, stopPos);
        this.stopTriggered = true;
        this.finishReason = "stop";
        this.drowseFinishReason = "stop_sequence";
        break;
      }
    }
    this.outputMessage = outputMessage;

    // Stop condition 3: exceed max_tokens
    if (!this.stopTriggered && this.outputIds.length >= max_tokens) {
      this.stopTriggered = true;
      this.finishReason = "length";
      this.drowseFinishReason = "length";
      log.info("Generation stopped due to exceeding max_tokens.");
    }

    // Stop condition 4: exceed KVCache's context window size
    if (
      !this.stopTriggered &&
      this.slidingWindowSize == -1 &&
      this.filledKVCacheLength == this.contextWindowSize
    ) {
      this.stopTriggered = true;
      this.finishReason = "length";
      this.drowseFinishReason = "length";
      log.info("Generation stopped due to exceeding context_window_size.");
    }

    // Finally, modify conversation history if stopped
    if (this.stopTriggered) {
      if (!this.conversation.isTextCompletion) {
        this.conversation.finishReply(this.outputPrefix + this.outputMessage);
      }
    }
  }

  /**
   * Given input tokens, return embeddings of them by calling embed kernel.
   *
   * @note precondition: inputTokens.length <= prefillChunkSize, since we take care of
   * chunking in `getChunkedPrefillInputData()`.
   */
  private getTokensEmbeddings(inputTokens: number[]): tvmjs.Tensor {
    this.tvm.beginScope();
    if (inputTokens.length > this.prefillChunkSize) {
      throw new Error(
        "Internal Error: getTokensEmbeddings input should be <= prefillChunkSize.",
      );
    }
    const inputData = this.tvm.empty(
      [inputTokens.length],
      "int32",
      this.device,
    );
    inputData.copyFrom(inputTokens);
    const embed: tvmjs.Tensor = this.tvm.detachFromCurrentScope(
      this.embed!(inputData, this.params),
    );
    this.tvm.endScope();
    this.tvm.attachToCurrentScope(embed); // tracked by scope of embedAndForward
    return embed;
  }

  /**
   * Compute the number of embedding tokens an image will produce.
   * Must match the model's image_embed output size.
   * Based on mlc_llm/serve/data.py _compute_embed_size.
   */
  private computeImageEmbedSize(
    imageHeight: number,
    imageWidth: number,
  ): number {
    const modelType = this.config.model_type;
    if (modelType === "phi3_v") {
      const [cropH, cropW] = this.calculateCropShape(imageHeight, imageWidth);
      const subTokens = cropH * 12 * (cropW * 12 + 1);
      const glbTokens = 12 * (12 + 1);
      return subTokens + 1 + glbTokens;
    }
    // For models with fixed embed size (e.g. Gemma3V)
    const mmTokens = this.config.model_config?.mm_tokens_per_image;
    if (mmTokens !== undefined) {
      return mmTokens;
    }
    throw new Error(
      `Cannot determine image embed size for model type "${modelType}". ` +
        "Please add mm_tokens_per_image to model_config in mlc-chat-config.json.",
    );
  }

  /**
   * Calculate resize dimensions per model type.
   * Based on vlm_utils.cc CalculateResizeShape
   */
  private calculateResizeShape(
    imageHeight: number,
    imageWidth: number,
  ): [number, number] {
    switch (this.config.model_type) {
      case "phi3_v": {
        const hdNum = 16;
        const ratio = imageWidth / imageHeight;
        let scale = 1;
        while (scale * Math.ceil(scale / ratio) <= hdNum) {
          scale += 1;
        }
        scale -= 1;
        const newW = scale * 336;
        const newH = Math.floor(newW / ratio);
        return [newH, newW];
      }
      default:
        throw new Error(
          `Unsupported model type "${this.config.model_type}" for image resize.`,
        );
    }
  }

  /**
   * Calculate crop dimensions per model type.
   * Based on vlm_utils.cc CalculateCropShape / CalculatePadShape
   */
  private calculateCropShape(
    imageHeight: number,
    imageWidth: number,
  ): [number, number] {
    switch (this.config.model_type) {
      case "phi3_v": {
        const [resizedHeight, resizedWidth] = this.calculateResizeShape(
          imageHeight,
          imageWidth,
        );
        const padH = Math.ceil(resizedHeight / 336) * 336;
        return [Math.floor(padH / 336), Math.floor(resizedWidth / 336)];
      }
      default:
        throw new Error(
          `Unsupported model type "${this.config.model_type}" for image crop.`,
        );
    }
  }

  /**
   * Embed an image input.
   */
  private async getImageEmbeddings(
    inputImage: ImageURL,
  ): Promise<tvmjs.Tensor> {
    this.tvm.beginScope();
    // 1. Transform ImageURL into image input in TVMArray
    const url = inputImage.url;
    const imgData: ImageData =
      this.imageDataCache.get(url) ?? (await getImageDataFromURL(url));
    const pixelValues: Uint8ClampedArray = getRGBArrayFromImageData(imgData);
    const pixelArray = this.tvm
      .empty([imgData.height, imgData.width, 3], "uint32", this.device)
      .copyFrom(pixelValues)
      .view([1, imgData.height, imgData.width, 3]); // NHWC

    // 2. Calculate resize and crop dimensions
    const [resizeH, resizeW] = this.calculateResizeShape(
      imgData.height,
      imgData.width,
    );
    const [cropH, cropW] = this.calculateCropShape(
      imgData.height,
      imgData.width,
    );
    const resizeHeightShape = this.tvm.makeShapeTuple([resizeH]);
    const resizeWidthShape = this.tvm.makeShapeTuple([resizeW]);
    const cropHeightShape = this.tvm.makeShapeTuple([cropH]);
    const cropWidthShape = this.tvm.makeShapeTuple([cropW]);

    // 3. embed kernel
    const embed: tvmjs.Tensor = this.tvm.detachFromCurrentScope(
      this.image_embed!(
        pixelArray,
        resizeHeightShape,
        resizeWidthShape,
        cropHeightShape,
        cropWidthShape,
        this.params,
      ),
    );
    const expectedSize = this.computeImageEmbedSize(
      imgData.height,
      imgData.width,
    );
    if (embed.shape[0] !== expectedSize) {
      throw new Error(
        `InternalError: expect embed.shape[0] to be ${expectedSize}, ` +
          `but got ${embed.shape[0]}`,
      );
    }
    this.tvm.endScope();
    this.tvm.attachToCurrentScope(embed); // tracked by scope of embedAndForward
    return embed;
  }

  /**
   * Embed and forward input data, that can be either array of tokens, or an image.
   * This will increment `this.filledKVCacheLength`.
   *
   * @param inputData data to embed and forward
   * @param inputDataLen length of this inputData, should smaller than prefill chunk size.
   * @returns The logits returned by this forward as tvmjs.Tensor on GPU.
   *
   * @note Precondition: inputData's data length is smaller than prefill chunk size
   */
  private async embedAndForward(
    inputData: Array<Array<number> | ImageURL>,
    inputDataLen: number,
    useDrowseProgram = true,
  ): Promise<tvmjs.Tensor> {
    if (
      useDrowseProgram &&
      (this.drowseProgram !== undefined ||
        this.drowseStructuredProgram !== undefined)
    ) {
      this.resetDrowseReadbackState();
    }
    if (inputDataLen > this.prefillChunkSize) {
      throw new Error(
        "InternalError: expect inputDataLen <= this.prefillChunkSize.",
      );
    }
    // TODO: we should combine string data to embed once, then rearrange the embeddings; currently
    // ["hi", imageUrl, "hi"] would call embed kernels 3 times, while 2 would suffice.

    // 1. Embed all inputData
    this.tvm.beginScope();
    const forwardStates = this.getActiveKVStates();
    let forwardBegun = 0;
    let logits: tvmjs.Tensor;
    try {
      const embeddings: tvmjs.Tensor[] = [];
      for (let i = 0; i < inputData.length; i++) {
        const data = inputData[i];
        if (Array.isArray(data)) {
          embeddings.push(this.getTokensEmbeddings(data));
        } else {
          embeddings.push(await this.getImageEmbeddings(data));
        }
      }

      // 2. Concatenate embeddings
      let allEmbeddings: tvmjs.Tensor;
      if (embeddings.length === 1) {
        allEmbeddings = embeddings[0];
      } else {
        allEmbeddings = this.tvm.concatEmbeddings(embeddings);
      }
      if (inputDataLen !== allEmbeddings.shape[0]) {
        throw new Error(
          "InternalError: expect seqLen == allEmbeddings.shape[0]",
        );
      }
      allEmbeddings = allEmbeddings.view([1].concat(allEmbeddings.shape));
      // TODO: Should we end this scope here and begin another scope? Will this dispose embeddings to
      // save RAM? We will detach allEmbeddings from this scope and attach to the next scope.

      // 3. Forward the concatenated embeddings
      const inputLenShape = this.tvm.makeShapeTuple([inputDataLen]);
      const seqIdsTuple = this.tvm.makeShapeTuple([0]);
      for (const state of forwardStates) {
        this.fKVCacheBeginForward!(state, seqIdsTuple, inputLenShape);
        forwardBegun += 1;
      }
      let retValue;
      if (inputDataLen > 1) {
        retValue = this.invokePrefill(
          allEmbeddings,
          inputDataLen,
          useDrowseProgram,
        );
      } else {
        retValue = this.invokeDecode(allEmbeddings, useDrowseProgram);
      }
      if (
        useDrowseProgram &&
        (this.drowseProgram !== undefined ||
          this.drowseStructuredProgram !== undefined)
      ) {
        const outputOffset = this.drowseOutputOffset();
        const measurements = this.tvm.detachFromCurrentScope(
          retValue.get(outputOffset),
        );
        this.drowseMeasurements?.dispose();
        this.drowseMeasurements = measurements;
        if (this.drowseStructuredMode === "curved") {
          const nextFeet = this.tvm.detachFromCurrentScope(
            retValue.get(outputOffset + 1),
          );
          this.drowseStructuredProgram!.curveFeet!.dispose();
          this.drowseStructuredProgram!.curveFeet = nextFeet;
        }
        const structured = this.drowseStructuredProgram;
        if (structured !== undefined) {
          const hiddenIndex =
            outputOffset + (this.drowseStructuredMode === "curved" ? 2 : 1);
          const hiddenStates = retValue.get(hiddenIndex) as tvmjs.Tensor;
          if (structured.geometryHeader !== undefined) {
            const geometryForward =
              inputDataLen > 1
                ? this.drowseGeometryPrefill!
                : this.drowseGeometryDecoding!;
            const geometry = geometryForward(
              hiddenStates,
              ...this.getDrowseGeometryArguments(),
              this.params,
            );
            const geometryMeasurements = this.tvm.detachFromCurrentScope(
              geometry.get(0),
            );
            const nextFeet = this.tvm.detachFromCurrentScope(geometry.get(1));
            this.drowseGeometryMeasurements?.dispose();
            this.drowseGeometryMeasurements = geometryMeasurements;
            structured.geometryFeet!.dispose();
            structured.geometryFeet = nextFeet;
          }
          if (this.drowseJlensChunks !== undefined) {
            await this.computeDrowseJlensProbabilities(
              hiddenStates,
              structured.jLensTokenIds!,
            );
          }
          await this.computeDrowseSaeTopFeatures(hiddenStates);
        }
      }

      // Epilogue
      while (forwardBegun > 0) {
        this.fKVCacheEndForward!(forwardStates[--forwardBegun]);
      }
      this.filledKVCacheLength += inputDataLen;
      logits = this.tvm.detachFromCurrentScope(retValue.get(0));
    } finally {
      try {
        while (forwardBegun > 0) {
          this.fKVCacheEndForward!(forwardStates[--forwardBegun]);
        }
      } finally {
        this.tvm.endScope();
      }
    }
    this.tvm.attachToCurrentScope(logits);
    return logits;
  }

  private static loadVMFunctionRegistry(
    vm: tvmjs.VirtualMachine,
    names: string[],
  ): VMFunctionRegistry {
    const registry: VMFunctionRegistry = {};
    const legacyNames = names
      .filter((name) => name.startsWith("drowse_"))
      .flatMap((name) =>
        ["polythetic_", "saklas_"].map(
          (prefix) => `${prefix}${name.slice("drowse_".length)}`,
        ),
      );
    for (const name of [...names, ...legacyNames]) {
      try {
        const func = vm.getFunction(name) as unknown;
        if (typeof func === "function") {
          registry[name] = func as tvmjs.PackedFunc;
        }
      } catch {
        // no-op for unexported symbols
      }
    }
    return registry;
  }

  private static applyLegacyDrowseAliases(registry: VMFunctionRegistry): void {
    for (const [name, func] of Object.entries(registry)) {
      const prefix = ["polythetic_", "saklas_"].find((value) =>
        name.startsWith(value),
      );
      if (prefix === undefined || func === undefined) continue;
      const currentName = `drowse_${name.slice(prefix.length)}`;
      if (registry[currentName] === undefined) registry[currentName] = func;
    }
  }

  private static getRequiredVMFunctionByName(
    name: string,
    registry: VMFunctionRegistry,
  ): tvmjs.PackedFunc {
    const func = registry[name];
    if (func !== undefined) {
      return func;
    }

    const availableNames = Object.entries(registry)
      .filter((entry) => entry[1] !== undefined)
      .map((entry) => entry[0])
      .sort();
    const availableStr =
      availableNames.length === 0 ? "(none)" : availableNames.join(", ");
    throw new Error(
      `Cannot find required VM function \`${name}\`. Available candidate functions: ${availableStr}`,
    );
  }

  private static getVMFunctionAvailability(
    registry: VMFunctionRegistry,
  ): VMFunctionAvailability {
    return {
      prefill: registry.prefill !== undefined,
      batch_prefill: registry.batch_prefill !== undefined,
      decode: registry.decode !== undefined,
      batch_decode: registry.batch_decode !== undefined,
      create_tir_paged_kv_cache:
        registry.create_tir_paged_kv_cache !== undefined,
      create_rnn_state: registry.create_rnn_state !== undefined,
    };
  }

  private parseKVStateKind(kvStateKindRaw: unknown): KVStateKind {
    if (kvStateKindRaw === undefined || kvStateKindRaw === null) {
      return "kv_cache";
    }
    if (typeof kvStateKindRaw !== "string") {
      throw new Error(
        `Invalid kv_state_kind in model metadata: expected string, got ${typeof kvStateKindRaw}`,
      );
    }
    const kvStateKind = kvStateKindRaw as KVStateKind;
    if (
      kvStateKind === "kv_cache" ||
      kvStateKind === "rnn_state" ||
      kvStateKind === "hybrid" ||
      kvStateKind === "none"
    ) {
      return kvStateKind;
    }
    throw new Error(
      `Unsupported kv_state_kind in model metadata: ${kvStateKindRaw}`,
    );
  }

  private static resolveModelABI(
    kvStateKind: KVStateKind,
    availability: VMFunctionAvailability,
    preferBatch = false,
  ): ResolvedModelABI {
    const hasSingleKernelPair = availability.prefill && availability.decode;
    const hasBatchKernelPair =
      availability.batch_prefill && availability.batch_decode;
    const availableNames = Object.entries(availability)
      .filter((entry) => entry[1])
      .map((entry) => entry[0])
      .sort();
    const availableStr =
      availableNames.length === 0 ? "(none)" : availableNames.join(", ");

    if (kvStateKind === "none") {
      throw new Error(
        "kv_state_kind=`none` is not supported in LLMChatPipeline chat runtime.",
      );
    }

    if (kvStateKind === "hybrid") {
      const missingFunctions: string[] = [];
      if (!availability.batch_prefill) {
        missingFunctions.push("batch_prefill");
      }
      if (!availability.batch_decode) {
        missingFunctions.push("batch_decode");
      }
      if (!availability.create_tir_paged_kv_cache) {
        missingFunctions.push("create_tir_paged_kv_cache");
      }
      if (!availability.create_rnn_state) {
        missingFunctions.push("create_rnn_state");
      }
      if (missingFunctions.length !== 0) {
        throw new Error(
          "Invalid hybrid ABI. Missing required functions: " +
            `${missingFunctions.join(", ")}. ` +
            `Available candidate functions: ${availableStr}`,
        );
      }
      return {
        kvStateKind,
        prefillABI: "batch",
        decodeABI: "batch",
        prefillFunctionName: "batch_prefill",
        decodeFunctionName: "batch_decode",
        needsKVCache: true,
        needsRNNState: true,
      };
    }

    if (kvStateKind === "rnn_state") {
      if (!availability.create_rnn_state) {
        throw new Error(
          "Invalid rnn_state ABI. Missing required function: create_rnn_state.",
        );
      }
      if (hasSingleKernelPair) {
        return {
          kvStateKind,
          prefillABI: "single",
          decodeABI: "single",
          prefillFunctionName: "prefill",
          decodeFunctionName: "decode",
          needsKVCache: false,
          needsRNNState: true,
        };
      }
      if (hasBatchKernelPair) {
        return {
          kvStateKind,
          prefillABI: "batch",
          decodeABI: "batch",
          prefillFunctionName: "batch_prefill",
          decodeFunctionName: "batch_decode",
          needsKVCache: false,
          needsRNNState: true,
        };
      }
      throw new Error(
        "Invalid rnn_state ABI. Require either `prefill`+`decode` or " +
          "`batch_prefill`+`batch_decode`. " +
          `Available candidate functions: ${availableStr}`,
      );
    }

    // kv_cache
    if (preferBatch && hasBatchKernelPair) {
      return {
        kvStateKind,
        prefillABI: "batch",
        decodeABI: "batch",
        prefillFunctionName: "batch_prefill",
        decodeFunctionName: "batch_decode",
        needsKVCache: true,
        needsRNNState: false,
      };
    }
    if (hasSingleKernelPair) {
      return {
        kvStateKind,
        prefillABI: "single",
        decodeABI: "single",
        prefillFunctionName: "prefill",
        decodeFunctionName: "decode",
        needsKVCache: true,
        needsRNNState: false,
      };
    }
    if (hasBatchKernelPair) {
      return {
        kvStateKind,
        prefillABI: "batch",
        decodeABI: "batch",
        prefillFunctionName: "batch_prefill",
        decodeFunctionName: "batch_decode",
        needsKVCache: true,
        needsRNNState: false,
      };
    }
    throw new Error(
      "Invalid kv_cache ABI. Require either `prefill`+`decode` or " +
        "`batch_prefill`+`batch_decode`. " +
        `Available candidate functions: ${availableStr}`,
    );
  }

  private requireKVCache(): tvmjs.TVMObject {
    if (this.kvCache === undefined) {
      throw new Error("InternalError: kv cache is not initialized.");
    }
    return this.kvCache;
  }

  private requireRNNState(): tvmjs.TVMObject {
    if (this.rnnState === undefined) {
      throw new Error("InternalError: rnn state is not initialized.");
    }
    return this.rnnState;
  }

  private getSingleStateForABI(): tvmjs.TVMObject {
    if (
      this.resolvedModelABI.needsKVCache &&
      !this.resolvedModelABI.needsRNNState
    ) {
      return this.requireKVCache();
    }
    if (
      !this.resolvedModelABI.needsKVCache &&
      this.resolvedModelABI.needsRNNState
    ) {
      return this.requireRNNState();
    }
    throw new Error(
      "InternalError: single-state ABI requested for a hybrid/non-single state model.",
    );
  }

  private getActiveKVStates(): tvmjs.TVMObject[] {
    const states: tvmjs.TVMObject[] = [];
    if (this.kvCache !== undefined) {
      states.push(this.kvCache);
    }
    if (this.rnnState !== undefined) {
      states.push(this.rnnState);
    }
    if (states.length === 0) {
      throw new Error("InternalError: no kv states initialized.");
    }
    return states;
  }

  private drowseStateArguments(): tvmjs.TVMObject[] {
    return [
      ...(this.resolvedModelABI.needsKVCache ? [this.requireKVCache()] : []),
      ...(this.resolvedModelABI.needsRNNState ? [this.requireRNNState()] : []),
    ];
  }

  private drowseOutputOffset(): number {
    return (
      1 +
      Number(this.resolvedModelABI.needsKVCache) +
      Number(this.resolvedModelABI.needsRNNState)
    );
  }

  private resetDrowseCaptureState(): void {
    this.tvm.beginScope();
    try {
      this.resetKVCache();
      this.filledKVCacheLength = 0;
    } finally {
      this.tvm.endScope();
    }
  }

  private async captureDrowseTokenChunk(
    inputIds: number[],
    positions: number[],
  ): Promise<Float32Array> {
    this.tvm.beginScope();
    const forwardStates = this.getActiveKVStates();
    let forwardBegun = 0;
    try {
      const embeddings = this.getTokensEmbeddings(inputIds);
      const allEmbeddings = embeddings.view([1].concat(embeddings.shape));
      const capturePositions = this.tvm
        .empty([positions.length], "int32", this.device)
        .copyFrom(new Int32Array(positions));
      const inputLenShape = this.tvm.makeShapeTuple([inputIds.length]);
      const seqIdsTuple = this.tvm.makeShapeTuple([0]);
      for (const state of forwardStates) {
        this.fKVCacheBeginForward(state, seqIdsTuple, inputLenShape);
        forwardBegun += 1;
      }
      const retValue = this.invokeDrowseCapture(
        allEmbeddings,
        inputIds.length,
        capturePositions,
      );
      const captures = retValue.get(this.drowseOutputOffset()) as tvmjs.Tensor;
      const layerCount = this.requireModelDimension("num_hidden_layers");
      const hiddenSize = this.requireModelDimension("hidden_size");
      const expectedShape = [layerCount, positions.length, hiddenSize];
      if (
        captures.dtype !== "float32" ||
        captures.shape.length !== expectedShape.length ||
        captures.shape.some((value, index) => value !== expectedShape[index])
      ) {
        throw new Error(
          `Drowse residual capture returned ${captures.dtype}[${captures.shape.join(",")}], ` +
            `expected float32[${expectedShape.join(",")}]`,
        );
      }
      for (let index = forwardStates.length - 1; index >= 0; index -= 1) {
        this.fKVCacheEndForward(forwardStates[index]);
      }
      forwardBegun = 0;
      this.filledKVCacheLength += inputIds.length;
      const host = this.tvm.empty(captures.shape, "float32", this.tvm.cpu());
      host.copyFrom(captures);
      await this.device.sync();
      return new Float32Array(host.toArray());
    } finally {
      if (forwardBegun) {
        for (let index = forwardBegun - 1; index >= 0; index -= 1) {
          this.fKVCacheEndForward(forwardStates[index]);
        }
      }
      this.tvm.endScope();
    }
  }

  private async captureDrowseRankOneTokenChunkV1(
    inputIds: number[],
    positions: number[],
  ): Promise<Float32Array> {
    this.tvm.beginScope();
    const forwardStates = this.getActiveKVStates();
    let forwardBegun = 0;
    try {
      const embeddings = this.getTokensEmbeddings(inputIds);
      const allEmbeddings = embeddings.view([1].concat(embeddings.shape));
      const capturePositions = this.tvm
        .empty([positions.length], "int32", this.device)
        .copyFrom(new Int32Array(positions));
      const inputLenShape = this.tvm.makeShapeTuple([inputIds.length]);
      const seqIdsTuple = this.tvm.makeShapeTuple([0]);
      for (const state of forwardStates) {
        this.fKVCacheBeginForward(state, seqIdsTuple, inputLenShape);
        forwardBegun += 1;
      }
      const retValue = this.invokeDrowseRankOneCaptureV1(
        allEmbeddings,
        inputIds.length,
        capturePositions,
      );
      const measurements = this.tvm.detachFromCurrentScope(
        retValue.get(this.drowseOutputOffset()) as tvmjs.Tensor,
      );
      this.drowseMeasurements?.dispose();
      this.drowseMeasurements = measurements;
      this.drowseMeasurementBundleHost = undefined;
      const captures = retValue.get(
        this.drowseOutputOffset() + 1,
      ) as tvmjs.Tensor;
      const layerCount = this.requireModelDimension("num_hidden_layers");
      const hiddenSize = this.requireModelDimension("hidden_size");
      const expectedShape = [layerCount, positions.length, hiddenSize];
      if (
        captures.dtype !== "float32" ||
        captures.shape.length !== expectedShape.length ||
        captures.shape.some((value, index) => value !== expectedShape[index])
      ) {
        throw new Error(
          `Drowse rank-one residual capture returned ${captures.dtype}[${captures.shape.join(",")}], ` +
            `expected float32[${expectedShape.join(",")}]`,
        );
      }
      for (let index = forwardStates.length - 1; index >= 0; index -= 1) {
        this.fKVCacheEndForward(forwardStates[index]);
      }
      forwardBegun = 0;
      this.filledKVCacheLength += inputIds.length;
      const host = this.tvm.empty(captures.shape, "float32", this.tvm.cpu());
      host.copyFrom(captures);
      await this.device.sync();
      return new Float32Array(host.toArray());
    } finally {
      if (forwardBegun) {
        for (let index = forwardBegun - 1; index >= 0; index -= 1) {
          this.fKVCacheEndForward(forwardStates[index]);
        }
      }
      this.tvm.endScope();
    }
  }

  private invokeDrowseCapture(
    allEmbeddings: tvmjs.Tensor,
    inputDataLen: number,
    capturePositions: tvmjs.Tensor,
  ): any {
    const isPrefill = inputDataLen > 1;
    const forward = isPrefill
      ? this.drowseCapturePrefill!
      : this.drowseCaptureDecoding!;
    const abi = isPrefill
      ? this.resolvedModelABI.prefillABI
      : this.resolvedModelABI.decodeABI;
    if (abi === "single") {
      return forward(
        allEmbeddings,
        ...this.drowseStateArguments(),
        capturePositions,
        this.params,
      );
    }
    if (isPrefill) {
      if (this.prefillLogitPositions === undefined) {
        throw new Error(
          "InternalError: batch prefill ABI requires prefillLogitPositions tensor.",
        );
      }
      this.prefillLogitPositionHost[0] = inputDataLen - 1;
      this.prefillLogitPositions.copyFrom(this.prefillLogitPositionHost);
      return forward(
        allEmbeddings,
        this.prefillLogitPositions,
        ...this.drowseStateArguments(),
        capturePositions,
        this.params,
      );
    }
    return forward(
      allEmbeddings,
      ...this.drowseStateArguments(),
      capturePositions,
      this.params,
    );
  }

  private invokeDrowseRankOneCaptureV1(
    allEmbeddings: tvmjs.Tensor,
    inputDataLen: number,
    capturePositions: tvmjs.Tensor,
  ): any {
    const isPrefill = inputDataLen > 1;
    const forward = isPrefill
      ? this.drowseRankOneCapturePrefillV1!
      : this.drowseRankOneCaptureDecodingV1!;
    const abi = isPrefill
      ? this.resolvedModelABI.prefillABI
      : this.resolvedModelABI.decodeABI;
    const drowseArgs = this.getDrowseForwardArguments();
    if (drowseArgs.length === 0) {
      throw new Error("Drowse rank-one capture requires an installed program");
    }
    if (abi === "single") {
      return forward(
        allEmbeddings,
        ...this.drowseStateArguments(),
        capturePositions,
        ...drowseArgs,
        this.params,
      );
    }
    if (isPrefill) {
      if (this.prefillLogitPositions === undefined) {
        throw new Error(
          "InternalError: batch prefill ABI requires prefillLogitPositions tensor.",
        );
      }
      this.prefillLogitPositionHost[0] = inputDataLen - 1;
      this.prefillLogitPositions.copyFrom(this.prefillLogitPositionHost);
      return forward(
        allEmbeddings,
        this.prefillLogitPositions,
        ...this.drowseStateArguments(),
        capturePositions,
        ...drowseArgs,
        this.params,
      );
    }
    return forward(
      allEmbeddings,
      ...this.drowseStateArguments(),
      capturePositions,
      ...drowseArgs,
      this.params,
    );
  }

  private invokePrefill(
    allEmbeddings: tvmjs.Tensor,
    inputDataLen: number,
    useDrowseProgram = true,
  ): any {
    let forward = this.prefill;
    if (useDrowseProgram && this.drowseProgram !== undefined) {
      forward = this.drowsePrefill!;
    } else if (useDrowseProgram && this.drowseStructuredMode === "structured") {
      forward = this.drowseStructuredPrefill!;
    } else if (useDrowseProgram && this.drowseStructuredMode === "curved") {
      forward = this.drowseCurvedPrefill!;
    }
    const drowseArgs = useDrowseProgram ? this.getDrowseForwardArguments() : [];
    if (this.resolvedModelABI.prefillABI === "single") {
      return forward(
        allEmbeddings,
        this.getSingleStateForABI(),
        ...drowseArgs,
        this.params,
      );
    }

    if (this.prefillLogitPositions === undefined) {
      throw new Error(
        "InternalError: batch prefill ABI requires prefillLogitPositions tensor.",
      );
    }
    this.prefillLogitPositionHost[0] = inputDataLen - 1;
    this.prefillLogitPositions.copyFrom(this.prefillLogitPositionHost);

    if (
      this.resolvedModelABI.needsKVCache &&
      this.resolvedModelABI.needsRNNState
    ) {
      return forward(
        allEmbeddings,
        this.prefillLogitPositions,
        this.requireKVCache(),
        this.requireRNNState(),
        ...drowseArgs,
        this.params,
      );
    }
    return forward(
      allEmbeddings,
      this.prefillLogitPositions,
      this.getSingleStateForABI(),
      ...drowseArgs,
      this.params,
    );
  }

  private invokeDecode(
    allEmbeddings: tvmjs.Tensor,
    useDrowseProgram = true,
  ): any {
    let forward = this.decoding;
    if (useDrowseProgram && this.drowseProgram !== undefined) {
      forward = this.drowseDecoding!;
    } else if (useDrowseProgram && this.drowseStructuredMode === "structured") {
      forward = this.drowseStructuredDecoding!;
    } else if (useDrowseProgram && this.drowseStructuredMode === "curved") {
      forward = this.drowseCurvedDecoding!;
    }
    const drowseArgs = useDrowseProgram ? this.getDrowseForwardArguments() : [];
    if (this.resolvedModelABI.decodeABI === "single") {
      return forward(
        allEmbeddings,
        this.getSingleStateForABI(),
        ...drowseArgs,
        this.params,
      );
    }
    if (
      this.resolvedModelABI.needsKVCache &&
      this.resolvedModelABI.needsRNNState
    ) {
      return forward(
        allEmbeddings,
        this.requireKVCache(),
        this.requireRNNState(),
        ...drowseArgs,
        this.params,
      );
    }
    return forward(
      allEmbeddings,
      this.getSingleStateForABI(),
      ...drowseArgs,
      this.params,
    );
  }

  private getDrowseForwardArguments(): tvmjs.Tensor[] {
    if (this.drowseProgram !== undefined) {
      return [
        this.drowseProgram.enabled,
        this.drowseProgram.basis,
        this.drowseProgram.neutral,
        this.drowseProgram.target,
        this.drowseProgram.along,
        this.drowseProgram.collapse,
        this.drowseProgram.probeBasis,
        this.drowseProgram.probeNeutral,
      ];
    }
    const program = this.drowseStructuredProgram;
    if (program === undefined) return [];
    const affine = [
      program.affineActive,
      program.affineBasis,
      program.affineNeutral,
      program.affineTarget,
      program.affineAlong,
      program.affineKappa,
      program.probeKind,
      program.probeDirection,
      program.probeBias,
      program.probeThreshold,
    ];
    if (this.drowseStructuredMode !== "curved") return affine;
    return affine.concat([
      program.curveBasis!,
      program.curveNeutral!,
      program.curveDomainKind!,
      program.curveParameters!,
      program.curveFeet!,
    ]);
  }

  private getDrowseGeometryArguments(): tvmjs.Tensor[] {
    const program = this.drowseStructuredProgram;
    if (program?.geometryHeader === undefined) return [];
    return [
      program.whitenerRank!,
      program.whitenerRidge!,
      program.whitenerBasis!,
      program.whitenerCorrection!,
      program.geometryHeader,
      program.geometryPayload!,
      program.geometryFeet!,
    ];
  }

  // NOTE: caller must call device.sync()
  private updateLogitsOnCPU(logits: tvmjs.Tensor): tvmjs.Tensor {
    if (this.logitsOnCPU == undefined) {
      this.logitsOnCPU = this.tvm.detachFromCurrentScope(
        this.tvm.empty(logits.shape, logits.dtype, this.tvm.cpu()),
      );
    } else {
      if (logits.shape[0] != this.logitsOnCPU.shape[0]) {
        throw Error("We expect the size of logits to remain unchanged");
      }
    }
    this.logitsOnCPU.copyFrom(logits);
    return this.logitsOnCPU;
  }

  private async sampleTokenFromLogits(
    logitsOnGPU: tvmjs.Tensor,
    genConfig?: GenerationConfig,
  ) {
    // 0. Get sampling values and penalties, possibly overridden by genConfig.
    // Also load other genConfig items like logit_bias. Consume all fields of `genConfig` here.
    function _hasValue(value: any): boolean {
      // if we use `if value` directly, `value` being 0 evaluates to false, violating semantics
      return value !== undefined && value !== null;
    }
    let temperature: number = this.config.temperature;
    let top_p: number = this.config.top_p;
    let top_k = 0;
    let repetition_penalty: number = this.config.repetition_penalty;
    let frequency_penalty: number = this.config.frequency_penalty;
    let presence_penalty: number = this.config.presence_penalty;
    let logit_bias: Record<string, number> | undefined = undefined;
    let logprobs: boolean | undefined = undefined;
    let top_logprobs: number | undefined = undefined;
    let response_format: ResponseFormat | undefined = undefined;

    if (genConfig !== undefined) {
      if (_hasValue(genConfig.temperature)) {
        temperature = genConfig.temperature!;
      }
      if (_hasValue(genConfig.top_p)) {
        top_p = genConfig.top_p!;
      }
      if (_hasValue(genConfig.top_k)) {
        top_k = genConfig.top_k!;
      }
      // TODO: setting top_p to 1.0 by default might run into issues since
      // top_p masking in relax uses < instead of <=
      // Set default top_p to 1.0 if not set
      if (!_hasValue(top_p)) {
        top_p = 1.0;
      }
      if (_hasValue(genConfig.repetition_penalty)) {
        repetition_penalty = genConfig.repetition_penalty!;
      }
      if (_hasValue(genConfig.frequency_penalty)) {
        frequency_penalty = genConfig.frequency_penalty!;
      }
      if (_hasValue(genConfig.presence_penalty)) {
        presence_penalty = genConfig.presence_penalty!;
      }
      // If only one of frequency or presence penalty is set, make the other one 0.0
      if (_hasValue(frequency_penalty) && !_hasValue(presence_penalty)) {
        presence_penalty = 0.0;
      }
      if (_hasValue(presence_penalty) && !_hasValue(frequency_penalty)) {
        frequency_penalty = 0.0;
      }
      if (!_hasValue(frequency_penalty)) {
        frequency_penalty = 0.0;
      }
      if (!_hasValue(presence_penalty)) {
        presence_penalty = 0.0;
      }
      if (_hasValue(genConfig.logit_bias)) {
        logit_bias = genConfig.logit_bias!;
      }
      if (_hasValue(genConfig.logprobs)) {
        logprobs = genConfig.logprobs!;
      }
      if (_hasValue(genConfig.top_logprobs)) {
        top_logprobs = genConfig.top_logprobs!;
      }
      if (_hasValue(genConfig.response_format)) {
        response_format = genConfig.response_format!;
      }
    }
    // Check range validity
    if (!Number.isFinite(top_p) || top_p < 0 || top_p > 1) {
      throw new RangeError("top_p", 0, 1);
    }
    if (!Number.isSafeInteger(top_k) || top_k < 0) {
      throw new NonNegativeError("top_k");
    }
    if (!Number.isFinite(temperature)) {
      throw new TypeError("temperature must be finite");
    }
    if (!(repetition_penalty > 0)) {
      throw new MinValueError("repetition_penalty", 0);
    }
    if (!(frequency_penalty >= -2.0 && frequency_penalty <= 2.0)) {
      throw new RangeError("frequency_penalty", -2.0, 2.0);
    }
    if (!(presence_penalty >= -2.0 && presence_penalty <= 2.0)) {
      throw new RangeError("presence_penalty", -2.0, 2.0);
    }

    const outputTokenBegin = performance.now();
    const grammarConstrained =
      response_format?.type === "json_object" ||
      response_format?.type === "grammar" ||
      response_format?.type === "structural_tag";

    // 0. Update logitsOnGPU with on-GPU grammar bitmasking
    if (grammarConstrained) {
      const grammarBitmaskBegin = performance.now();

      this.tvm.beginScope();
      if (this.grammarMatcher === undefined) {
        throw Error("Expect grammar matcher to be initialized.");
      }

      const tBitmaskStart = performance.now();
      const bitMaskOnCPU: Int32Array =
        await this.grammarMatcher.getNextTokenBitmask();
      this.curRoundGrammarPerTokenTotalTime +=
        (performance.now() - tBitmaskStart) / 1e3;

      if (bitMaskOnCPU.length !== this.bitmaskSize) {
        throw new Error(
          `InternalError: Expect grammar bitmask to be ` +
            `size ${this.bitmaskSize}, but got ${bitMaskOnCPU.length}.`,
        );
      }
      const bitMaskOnGPU = this.tvm
        .empty([1, this.bitmaskSize], "int32", this.device)
        .copyFrom(bitMaskOnCPU);
      const seqIdsArray = this.tvm
        .empty([1], "int32", this.device)
        .copyFrom([0]);
      this.fapplyBitmask(
        logitsOnGPU.view([1, this.fullVocabSize]),
        seqIdsArray,
        bitMaskOnGPU,
      );
      this.tvm.endScope();

      if (genConfig?.enable_latency_breakdown) {
        const grammarBitmaskEnd = performance.now();
        const grammarBitmaskTimeSpent =
          (grammarBitmaskEnd - grammarBitmaskBegin) / 1e3;
        this.curRoundLatencyBreakdown.grammarBitmaskTime.push(
          grammarBitmaskTimeSpent,
        );
      }
    }

    // 1. Apply logitProcessor on CPU
    if (this.logitProcessor !== undefined) {
      // Move logits to CPU
      this.tvm.beginScope();
      this.updateLogitsOnCPU(logitsOnGPU);
      this.tvm.endScope();
      await this.device.sync();

      const logitProcessorBegin = performance.now();

      if (this.logitsOnCPU == undefined) {
        throw Error("logits should be assigned");
      }
      let logitsOnCPUArray: Float32Array = <Float32Array>(
        this.logitsOnCPU.toArray()
      );
      logitsOnCPUArray = this.logitProcessor.processLogits(logitsOnCPUArray);
      logitsOnGPU.copyFrom(logitsOnCPUArray);
      this.logitsOnCPU.copyFrom(logitsOnCPUArray);

      if (genConfig?.enable_latency_breakdown) {
        const logitProcessorEnd = performance.now();
        const logitProcessorTimeSpent =
          (logitProcessorEnd - logitProcessorBegin) / 1e3;
        this.curRoundLatencyBreakdown.logitProcessorTime.push(
          logitProcessorTimeSpent,
        );
      }
    }

    // 2. Apply logit_bias on GPU
    if (_hasValue(logit_bias)) {
      const logitBiasBegin = performance.now();

      const numTokens = Object.keys(logit_bias ?? {}).length;
      const pos2seqIds = new Int32Array(numTokens).fill(0);
      const tokenIds = new Int32Array(numTokens);
      const tokenLogitBias = new Float32Array(numTokens);

      const logitBiasKeys = Object.keys(logit_bias ?? {});
      for (let index = 0; index < numTokens; index++) {
        const tokenId = parseInt(logitBiasKeys[index]);
        tokenIds[index] = tokenId;
        tokenLogitBias[index] = logit_bias![tokenId];
      }

      this.tvm.beginScope();

      const pos2seqIdsDevice = this.tvm
        .empty([numTokens], "int32", this.device)
        .copyFrom(pos2seqIds);

      const tokenIdsDevice = this.tvm
        .empty([numTokens], "int32", this.device)
        .copyFrom(tokenIds);

      const tokenLogitBiasDevice = this.tvm
        .empty([numTokens], "float32", this.device)
        .copyFrom(tokenLogitBias);

      this.fapplyLogitBias(
        logitsOnGPU.view([1, this.fullVocabSize]),
        pos2seqIdsDevice,
        tokenIdsDevice,
        tokenLogitBiasDevice,
      );

      this.tvm.endScope();

      if (genConfig?.enable_latency_breakdown) {
        const logitBiasEnd = performance.now();
        const logitBiasTimeSpent = (logitBiasEnd - logitBiasBegin) / 1e3;
        this.curRoundLatencyBreakdown.logitBiasTime.push(logitBiasTimeSpent);
      }
    }

    // 3. Apply penalties to logits on GPU
    if (
      frequency_penalty != 0.0 ||
      presence_penalty != 0.0 ||
      repetition_penalty != 1.0
    ) {
      const appearedTokens = [...this.appearedTokensFreq.keys()];
      const appearedTokensFreqs = [...this.appearedTokensFreq.values()];

      const numTokens = appearedTokens.length;

      if (numTokens > 0) {
        const penaltyBegin = performance.now();

        const pos2seqIds = new Int32Array(numTokens).fill(0);
        const tokenIds = new Int32Array(numTokens).fill(0);
        const tokenCnt = new Int32Array(numTokens).fill(0);
        const penalties = new Float32Array([
          presence_penalty,
          frequency_penalty,
          repetition_penalty,
        ]);

        tokenIds.set(appearedTokens);
        tokenCnt.set(appearedTokensFreqs);

        this.tvm.beginScope();
        const seqIdsArray = this.tvm
          .empty([1], "int32", this.device)
          .copyFrom([0]);

        const pos2seqIdsDevice = this.tvm
          .empty([numTokens], "int32", this.device)
          .copyFrom(pos2seqIds);

        const tokenIdsDevice = this.tvm
          .empty([numTokens], "int32", this.device)
          .copyFrom(tokenIds);

        const tokenCntDevice = this.tvm
          .empty([numTokens], "int32", this.device)
          .copyFrom(tokenCnt);

        const penaltiesDevice = this.tvm
          .empty([1, 3], "float32", this.device)
          .copyFrom(penalties);

        this.fapplyPenalty(
          logitsOnGPU.view([1, this.fullVocabSize]),
          seqIdsArray,
          pos2seqIdsDevice,
          tokenIdsDevice,
          tokenCntDevice,
          penaltiesDevice,
        );

        this.tvm.endScope();

        if (genConfig?.enable_latency_breakdown) {
          const penaltyEnd = performance.now();
          const penaltyTimeSpent = (penaltyEnd - penaltyBegin) / 1e3;
          this.curRoundLatencyBreakdown.penaltyTime.push(penaltyTimeSpent);
        }
      }
    }

    // TODO: Explore usage of multinomial sampling kernel (currently blocked due to usage
    // of i8) for cases where top_p is not set
    // 4. Sample token from logits
    const sampleBegin = performance.now();

    const forcedPrefix = genConfig?.drowse_forced_prefix_token_ids;
    const selectedTokenIds = genConfig?.drowse_score_token_ids ?? [];
    const replayRequested =
      (forcedPrefix !== undefined && forcedPrefix !== null) ||
      (genConfig?.drowse_score_token_ids !== undefined &&
        genConfig.drowse_score_token_ids !== null);
    for (const tokenId of [...(forcedPrefix ?? []), ...selectedTokenIds]) {
      if (tokenId >= this.fullVocabSize) {
        throw new Error(
          `Drowse replay token ID ${tokenId} is outside the vocabulary`,
        );
      }
    }

    const forcedToken =
      forcedPrefix !== undefined &&
      forcedPrefix !== null &&
      this.curRoundSampledTokens < forcedPrefix.length
        ? forcedPrefix[this.curRoundSampledTokens]
        : null;
    const samplerSelectedTokenIds = Array.from(
      new Set([
        ...selectedTokenIds,
        ...(forcedToken === null ? [] : [forcedToken]),
      ]),
    );
    const effectiveTopK = effectiveDrowseTopK(top_k, this.fullVocabSize);
    const deterministic = temperature <= 0 || top_p === 0;
    const samplingTemperature = deterministic ? 1 : temperature;
    const numSeqs = 1;
    const numProbs = 1;
    const temperatures = new Float32Array([samplingTemperature]);

    this.tvm.beginScope();
    const temperaturesDevice = this.tvm
      .empty([numSeqs], "float32", this.device)
      .copyFrom(temperatures);

    let probs = this.fsoftmaxWithTemperature(
      logitsOnGPU.view([numSeqs, numProbs, this.fullVocabSize]),
      temperaturesDevice,
    );
    probs = probs.view([numProbs, this.fullVocabSize]);

    const topPValue = deterministic ? 0 : top_p;
    const captureLogprobs = Boolean(logprobs) || replayRequested;
    const argsortResults = this.fargsortProbs(probs);
    const sortedProbsDevice = argsortResults.get(0);
    const sortedIndicesDevice = argsortResults.get(1);
    const sortedProbsHost = this.tvm
      .empty(sortedProbsDevice.shape, sortedProbsDevice.dtype, this.tvm.cpu())
      .copyFrom(sortedProbsDevice);
    const sortedIndicesHost = this.tvm
      .empty(
        sortedIndicesDevice.shape,
        sortedIndicesDevice.dtype,
        this.tvm.cpu(),
      )
      .copyFrom(sortedIndicesDevice);
    const uniformSamplesHost = deterministic
      ? undefined
      : this.tvm
          .empty([1], "float32", this.tvm.cpu())
          .copyFrom(this.tvm.uniform([1], 0.0, 1.0, this.device));
    await this.device.sync();
    const samplerResult = sampleDrowseTopKTopP(
      sortedProbsHost.toArray() as Float32Array,
      sortedIndicesHost.toArray() as Int32Array,
      topPValue,
      effectiveTopK,
      uniformSamplesHost === undefined
        ? 0
        : (uniformSamplesHost.toArray() as Float32Array)[0],
      samplerSelectedTokenIds,
    );
    const sampledToken = samplerResult.sampledTokenId;
    this.tvm.endScope();
    if (sampledToken < 0) {
      throw new Error("InternalError: failed to sample a valid token.");
    }
    if (genConfig?.enable_latency_breakdown) {
      const sampleEnd = performance.now();
      const sampleTimeSpent = (sampleEnd - sampleBegin) / 1e3;
      this.curRoundLatencyBreakdown.sampleTime.push(sampleTimeSpent);
    }

    const emittedToken = this.commitSampledToken(
      sampledToken,
      forcedToken,
      samplerResult,
      captureLogprobs,
      logprobs ? (top_logprobs ?? 0) : 0,
      grammarConstrained,
      replayRequested,
      selectedTokenIds,
    );

    if (genConfig?.enable_latency_breakdown) {
      const outputTokenEnd = performance.now();
      const outputTokenTimeSpent = (outputTokenEnd - outputTokenBegin) / 1e3;
      this.curRoundLatencyBreakdown.totalTime.push(outputTokenTimeSpent);
    }

    return emittedToken;
  }

  private commitSampledToken(
    sampledToken: number,
    forcedToken: number | null,
    samplerResult: DrowseSamplerResult,
    captureLogprobs: boolean,
    topLogprobs: number,
    grammarConstrained: boolean,
    replayRequested: boolean,
    selectedTokenIds: readonly number[],
  ): number {
    const emittedToken = forcedToken ?? sampledToken;
    const selected = new Set(selectedTokenIds);
    const replayMetadata: DrowseReplayTokenMetadata | undefined =
      replayRequested
        ? {
            emitted_token_id: emittedToken,
            sampled_token_id: sampledToken,
            forced_token_id: forcedToken,
            selected_logprobs: samplerResult.selectedLogprobs.filter((row) =>
              selected.has(row.token_id),
            ),
            argmax: samplerResult.argmax,
            top_logprobs: samplerResult.topLogprobs,
          }
        : undefined;
    if (captureLogprobs) {
      this.tokenLogprobArray.push(
        this.getTokenLogprob(
          emittedToken,
          topLogprobs,
          samplerResult,
          replayMetadata,
        ),
      );
    }

    this.logitProcessor?.processSampledToken(emittedToken);
    if (grammarConstrained) {
      if (this.grammarMatcher === undefined) {
        throw Error("Expect grammar matcher to be initialized.");
      }
      const tAcceptStart = performance.now();
      const accepted = this.grammarMatcher.acceptToken(emittedToken);
      this.curRoundGrammarPerTokenTotalTime +=
        (performance.now() - tAcceptStart) / 1e3;
      if (!accepted) {
        throw Error("Grammar matcher rejected the newly sampled token.");
      }
    }
    this.curRoundSampledTokens += 1;
    return emittedToken;
  }

  /**
   * Return the an array of a mixture of token arrays and imageURLs (which cannot be represented
   * as tokens). Also return the number of tokens this represents.
   *
   * We first convert the Conversation into a prompt array to be prefilled. Then we encode the
   * text parts, leaving the imageURLs as it is.
   * Example prompts:
   * [
   *   "<|system|>\nSome system prompt\n",
   *   [
   *     "<|user|>\n",
   *     imageURL1,
   *     "\n",
   *     imageURL2,
   *     "\n",
   *     "Some user input<|end|>\n"
   *   ],
   * ]
   *
   * Expected output:
   * [
   *   token array for "<|system|>\nSome system prompt\n<|user|>\n",
   *   imageUrl1,
   *   token array for "\n",
   *   imageUrl2,
   *   token array for "\nSome user input<|end|>\n"
   */
  private async getInputData(): Promise<
    [Array<Array<number> | ImageURL>, number, (image: ImageURL) => number]
  > {
    const ret: Array<Array<number> | ImageURL> = [];
    let curTokens: Array<number> = [];
    let prompts: Array<string | Array<string | ImageURL>>;

    // 1. Get prompts
    if (this.conversation.isTextCompletion) {
      // 1.1. Non-conversation style
      if (this.filledKVCacheLength !== 0) {
        throw new TextCompletionExpectsKVEmptyError();
      }
      const legacyConfig = this.config as ChatConfig & Record<string, unknown>;
      const prefixes = ["drowse", "polythetic", "saklas"]
        .map((name) => legacyConfig[`${name}_completion_prefix_token_ids`])
        .filter((value) => value !== undefined);
      if (
        prefixes.some(
          (value) => JSON.stringify(value) !== JSON.stringify(prefixes[0]),
        )
      ) {
        throw new Error(
          "The completion-prefix metadata contains conflicting values",
        );
      }
      const prefix = (prefixes[0] ?? []) as number[];
      if (
        !Array.isArray(prefix) ||
        prefix.some(
          (id) =>
            !Number.isSafeInteger(id) || id < 0 || id >= this.fullVocabSize,
        )
      ) {
        throw new TypeError(
          "Text completion prefix must contain valid vocabulary token IDs",
        );
      }
      curTokens = [...prefix];
      prompts = this.conversation.getPromptArrayTextCompletion();
    } else {
      // 1.2. Conversation style
      if (this.filledKVCacheLength === 0) {
        if (
          this.conversation.config.system_prefix_token_ids !== undefined &&
          this.conversation.config.system_prefix_token_ids !== null
        ) {
          curTokens = [...this.conversation.config.system_prefix_token_ids];
        }
        prompts = this.conversation.getPromptArray(this.config);
      } else {
        prompts = this.conversation.getPromptArrayLastRound(this.config);
      }
    }

    // 1.5. Preload image dimensions to compute per-image embed sizes
    const imageDimensions = new Map<string, [number, number]>();
    const uniqueImageUrls = new Set<string>();
    for (const prompt of prompts) {
      if (typeof prompt !== "string") {
        for (const content of prompt) {
          if (typeof content !== "string") {
            uniqueImageUrls.add(content.url);
          }
        }
      }
    }
    if (uniqueImageUrls.size > 0 && this.image_embed === undefined) {
      throw new CannotFindImageEmbedError();
    }
    this.imageDataCache.clear();
    await Promise.all(
      Array.from(uniqueImageUrls).map(async (url) => {
        const imgData = await getImageDataFromURL(url);
        this.imageDataCache.set(url, imgData);
        imageDimensions.set(url, [imgData.height, imgData.width]);
      }),
    );
    const getEmbedSize = (image: ImageURL): number => {
      const dims = imageDimensions.get(image.url);
      if (!dims) {
        throw new Error("InternalError: image dimensions not preloaded");
      }
      return this.computeImageEmbedSize(dims[0], dims[1]);
    };

    // 2. Resolve BOI/EOI tokens
    const cfg = this.config.model_config;
    const boiToken = cfg?.vision_start_token_id ?? cfg?.boi_token_index;
    const eoiToken = cfg?.vision_end_token_id ?? cfg?.eoi_token_index;

    // 3. Encode all prompts. Iterate through each message in the prompt array, where each
    // prompt can either be a string, or an array of a mixture of string and ImageURLs.
    let numPromptTokens = curTokens.length;
    for (let i = 0; i < prompts.length; i++) {
      const curPrompt = prompts[i];
      if (typeof curPrompt === "string") {
        const encoded = this.tokenizer.encode(curPrompt);
        numPromptTokens += encoded.length;
        curTokens.push(...encoded);
      } else {
        for (let j = 0; j < curPrompt.length; j++) {
          const curPromptContent: string | ImageURL = curPrompt[j];
          if (typeof curPromptContent === "string") {
            const encoded = this.tokenizer.encode(curPromptContent);
            numPromptTokens += encoded.length;
            curTokens.push(...encoded);
          } else {
            // Insert BOI token before image if configured
            if (boiToken !== undefined) {
              curTokens.push(boiToken);
              numPromptTokens += 1;
            }
            // push curTokens to ret, push imageUrl, create a new curTokens
            ret.push([...curTokens]);
            ret.push(curPromptContent);
            numPromptTokens += getEmbedSize(curPromptContent);
            curTokens = [];
            // Insert EOI token after image if configured
            if (eoiToken !== undefined) {
              curTokens.push(eoiToken);
              numPromptTokens += 1;
            }
          }
        }
      }
    }
    // Deal with last curTokens
    if (curTokens.length !== 0) {
      ret.push([...curTokens]);
    }

    // Check if input tokens exceed context window size
    if (
      this.slidingWindowSize == -1 && // There is no limit on contextWindowSize for sliding window
      numPromptTokens + this.filledKVCacheLength > this.contextWindowSize
    ) {
      throw new ContextWindowSizeExceededError(
        numPromptTokens,
        this.contextWindowSize,
      );
    }
    return [ret, numPromptTokens, getEmbedSize];
  }

  async forwardTokensAndSample(
    inputIds: Array<number>,
    isPrefill: boolean,
  ): Promise<number> {
    const tstart = performance.now();
    this.tvm.beginScope();
    // 1. Chunk inputData if needed
    const inputData: Array<Array<number>> = [inputIds];
    const retGetChunks = getChunkedPrefillInputData(
      inputData,
      this.prefillChunkSize,
      () => 0, // text-only path, no images
    );
    const chunks: Array<Array<number> | ImageURL>[] = retGetChunks[0];
    const chunkLens: Array<number> = retGetChunks[1];

    // 2. Prefill each chunk
    let logitsOnGPU: tvmjs.Tensor;
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const chunkLen = chunkLens[i];
      const prevFilledLen = this.filledKVCacheLength;
      logitsOnGPU = await this.embedAndForward(chunk, chunkLen);
      if (this.filledKVCacheLength !== prevFilledLen + chunkLen) {
        throw new Error(
          "Internal Error: filledKVCacheLength does not match expected value.",
        );
      }
    }

    // 3. Sample next token
    const nextToken = await this.sampleTokenFromLogits(logitsOnGPU!);
    this.tvm.endScope();

    // 4. Stats
    const tend = performance.now();
    if (isPrefill) {
      // We assume that if the input has more than 1 token
      this.prefillTotalTime += (tend - tstart) / 1e3;
      this.prefillTotalTokens += inputIds.length;
      this.curRoundPrefillTotalTokens += inputIds.length;
      this.curRoundPrefillTotalTime += (tend - tstart) / 1e3;
    } else {
      this.decodingTotalTime += (tend - tstart) / 1e3;
      this.decodingTotalTokens += 1;
      this.curRoundDecodingTotalTokens += 1;
      this.curRoundDecodingTotalTime += (tend - tstart) / 1e3;
    }
    return nextToken;
  }

  /**
   * Based on `emittedToken` and `this.logitsOnCPU`, which becomes a distribution after
   * calling `this.tvm.applySoftmaxWithTemperature()`, generate `ChatCompletionTokenLogprob` and
   * update `this.tokenLogprobArray`.
   *
   * @param emittedToken The token ID emitted after any Drowse forced replay override.
   * @param top_logprobs Number of top tokens to include; `top_logprobs` in `ChatCompletionRequest`.
   *
   * @return The `ChatCompletionTokenLogprob` for this single autoregressive step.
   */
  private getTokenLogprob(
    emittedToken: number,
    top_logprobs: number,
    samplerResult: DrowseSamplerResult,
    drowseReplay?: DrowseReplayTokenMetadata,
  ): ChatCompletionTokenLogprob {
    // Get entry for sampled token first
    const textEncoder = new TextEncoder();
    const tokenStr = this.tokenizer.decode(new Int32Array([emittedToken]));
    const bytes: Array<number> = Array.from(textEncoder.encode(tokenStr));
    const logprob = samplerLogprobForToken(samplerResult, emittedToken);

    // Populate `top_logprobs`
    const topLogprobArray: Array<TopLogprob> = [];
    for (const row of samplerResult.topLogprobs.slice(0, top_logprobs)) {
      const tokenID_i = row.token_id;
      const tokenStr_i = this.tokenizer.decode(new Int32Array([tokenID_i]));
      topLogprobArray.push({
        token_id: tokenID_i,
        token: tokenStr_i,
        bytes: Array.from(textEncoder.encode(tokenStr_i)) as Array<number>,
        logprob: row.logprob,
      } as TopLogprob);
    }

    return {
      token_id: emittedToken,
      token: tokenStr,
      bytes: bytes,
      logprob: logprob,
      top_logprobs: topLogprobArray,
      drowse_sampler: {
        entropy_nats: samplerResult.entropyNats,
        perplexity: samplerResult.perplexity,
      },
      ...(drowseReplay === undefined ? {} : { drowse_replay: drowseReplay }),
    } as ChatCompletionTokenLogprob;
  }

  /**
   * Synchronize the device.
   */
  async sync(): Promise<void> {
    // Is it equivalent to this.tvm.sync()?
    await this.device.sync();
  }

  async evaluate() {
    // run a canonical evaluation of the flow
    this.resetKVCache();
    this.filledKVCacheLength = 0;

    const testPrompt = "The capital of Canada is";
    const ids = await this.tokenizer.encode(testPrompt);
    const tokens = Array.from(ids);
    tokens.unshift(this.bosTokenId);
    if (tokens.length == 0) {
      throw Error("empty token");
    }

    this.tvm.beginScope();
    const prefillChunk: Array<Array<number>> = [tokens] as Array<Array<number>>;
    const prefillChunkLen = tokens.length;
    const prefillStart = performance.now();
    await this.embedAndForward(prefillChunk, prefillChunkLen);
    this.tvm.endScope();
    await this.device.sync();

    const decodingStart = performance.now();

    this.tvm.beginScope();
    const decodeChunk: Array<Array<number>> = [[6234]];
    const decodeChunkLen = 1;
    const logitsOnCPU = this.updateLogitsOnCPU(
      await this.embedAndForward(decodeChunk, decodeChunkLen),
    );
    await this.device.sync();
    this.tvm.endScope();

    const decodingEnd = performance.now();
    const msg =
      `prefill-time=${((decodingStart - prefillStart) / 1000).toFixed(4)} sec` +
      `decoding-time=${((decodingEnd - decodingStart) / 1000).toFixed(4)} sec`;

    // simply log tokens for eyeballing.
    log.info("Logits:");
    log.info(logitsOnCPU.toArray());
    log.info(msg);
  }
}

function isTokenPrefix(prefix: number[], values: number[]): boolean {
  return (
    prefix.length <= values.length &&
    prefix.every((value, index) => value === values[index])
  );
}
