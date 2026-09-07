export const DROWSE_HOOK_ABI = "post-block-residual-v4" as const;
export const DROWSE_EXACT_READOUT_ABI = "exact-readout-v1" as const;

export interface DrowseRuntimeCapabilities {
  topK: boolean;
  forcedReplay: boolean;
  replayScoring: boolean;
  tokenizer: boolean;
  namedRoles: boolean;
  userSeatGeneration: boolean;
  sceneStitching: boolean;
}

export const DROWSE_READOUT_TOP_K = 8;

export interface DrowseSaeDictionary {
  hookAbi: typeof DROWSE_HOOK_ABI;
  bindingId: string;
  hiddenSize: number;
  runtimeLayerIndex: number;
  featureCount: number;
  activation: "relu" | "jump_relu";
  encoder: Float32Array;
  encoderBias: Float32Array;
  encoderThreshold: Float32Array;
  decoderBias: Float32Array;
}

export interface DrowseJlensDictionary {
  hookAbi: typeof DROWSE_HOOK_ABI;
  bindingId: string;
  hiddenSize: number;
  layerIndices: Int32Array;
  matrices: readonly Float32Array[];
}

export interface DrowseJlensTopTokenReadout {
  tokenIds: Int32Array;
  strength: Float32Array;
  centerOfMass: Float32Array;
  spread: Float32Array;
  fittedLayerCount: number;
  layerIndices: Int32Array;
  layerTokenIds: Int32Array;
  layerProbabilities: Float32Array;
}

export interface DrowseSaeTopFeatureReadout {
  featureIds: Int32Array;
  activations: Float32Array;
  runtimeLayerIndex: number;
  featureCount: number;
}

export interface DrowseMeasurementBundle {
  scalar?: Float32Array;
  geometry?: Float32Array;
  jlensTopTokens?: DrowseJlensTopTokenReadout;
  saeTopFeatures?: DrowseSaeTopFeatureReadout;
}

export interface DrowseStructuredHookProfile {
  schemaVersion: 3;
  id: "standard-v3";
  hookAbi: typeof DROWSE_HOOK_ABI;
  format: typeof DROWSE_STRUCTURED_HOOK_FORMAT;
  layerCount: number;
  hiddenSize: number;
  maxAffineGroups: number;
  maxRank: number;
  maxProbes: number;
  maxCurves: number;
  maxCurveNodes: number;
  maxIntrinsicDim: number;
  maxEmbedDim: number;
  curveParameterStride: number;
  maxComputeWorkgroupStorageSize: number;
  maxGeometryProbes: number;
  maxWhitenerRank: number;
  maxGeometryCandidates: number;
  geometryOutputStride: number;
  geometryFootRestarts: number;
  geometryFootIterations: number;
  geometryWarmRestarts: number;
  geometryWarmIterations: number;
  requiredMaxStorageBuffersPerShaderStage: number;
  geometryKernelStorageBindings: number;
  geometryHeaderStride: number;
  geometryPayloadElements: number;
  exactReadoutAbiVersion: 1;
  exactReadoutAbi: typeof DROWSE_EXACT_READOUT_ABI;
  readoutTopK: number;
  maxSaeFeaturesPerChunk: number;
}

export interface DrowseRankOneProgram {
  hookAbi: typeof DROWSE_HOOK_ABI;
  hiddenSize: number;
  layerCount: number;
  enabled: Uint32Array;
  basis: Float32Array;
  neutral: Float32Array;
  target: Float32Array;
  along: Float32Array;
  collapse: Float32Array;
  probeBasis: Float32Array;
  probeNeutral: Float32Array;
}

export const DROWSE_STRUCTURED_HOOK_FORMAT_V2 = "drowse-structured-v2" as const;
export const DROWSE_STRUCTURED_HOOK_FORMAT = "drowse-structured-v3" as const;
export type DrowseStructuredHookFormat =
  | typeof DROWSE_STRUCTURED_HOOK_FORMAT_V2
  | typeof DROWSE_STRUCTURED_HOOK_FORMAT;
export const DROWSE_STRUCTURED_MAX_AFFINE_GROUPS = 4;
export const DROWSE_STRUCTURED_MAX_RANK = 8;
export const DROWSE_STRUCTURED_MAX_PROBES = 8;
export const DROWSE_STRUCTURED_MAX_CURVES = 4;
export const DROWSE_STRUCTURED_MAX_CURVE_NODES = 32;
export const DROWSE_STRUCTURED_MAX_INTRINSIC_DIM = 4;
export const DROWSE_STRUCTURED_MAX_EMBED_DIM = 8;
export const DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE = 671;
export const DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES = 8;
export const DROWSE_STRUCTURED_MAX_WHITENER_RANK = 96;
export const DROWSE_STRUCTURED_MAX_GEOMETRY_CANDIDATES = 33;
export const DROWSE_STRUCTURED_GEOMETRY_OUTPUT_STRIDE = 41;
export const DROWSE_STRUCTURED_GEOMETRY_FOOT_RESTARTS = 3;
export const DROWSE_STRUCTURED_GEOMETRY_FOOT_ITERATIONS = 12;
export const DROWSE_STRUCTURED_GEOMETRY_WARM_RESTARTS = 2;
export const DROWSE_STRUCTURED_GEOMETRY_WARM_ITERATIONS = 4;
export const DROWSE_STRUCTURED_GEOMETRY_HEADER_STRIDE = 7;
export const DROWSE_STRUCTURED_GEOMETRY_KERNEL_STORAGE_BINDINGS = 8;
export const DROWSE_STRUCTURED_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE = 8;
export const DROWSE_STRUCTURED_PROFILE_SCHEMA_VERSION = 3 as const;
export const DROWSE_STRUCTURED_PROFILE_ID = "standard-v3" as const;
export const DROWSE_STRUCTURED_PROFILE_MAGIC = 0x53414b4c;
export const DROWSE_STRUCTURED_PROFILE_DESCRIPTOR_LENGTH = 31;
export const DROWSE_EXACT_READOUT_ABI_VERSION = 1 as const;
export const DROWSE_EXACT_READOUT_MAX_SAE_FEATURES_PER_CHUNK = 16384;
export const DROWSE_STRUCTURED_MIN_COMPUTE_WORKGROUP_STORAGE_SIZE = 32 * 1024;

export function drowseGeometryPayloadElements(
  layerCount: number,
  hiddenSize: number,
): number {
  const slots = layerCount * DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES;
  return (
    slots *
    (2 * hiddenSize +
      DROWSE_STRUCTURED_MAX_RANK * hiddenSize +
      2 * DROWSE_STRUCTURED_MAX_RANK * DROWSE_STRUCTURED_MAX_RANK +
      DROWSE_STRUCTURED_MAX_GEOMETRY_CANDIDATES * DROWSE_STRUCTURED_MAX_RANK +
      DROWSE_STRUCTURED_MAX_INTRINSIC_DIM * DROWSE_STRUCTURED_MAX_RANK +
      DROWSE_STRUCTURED_MAX_INTRINSIC_DIM +
      DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE +
      DROWSE_STRUCTURED_MAX_CURVE_NODES * DROWSE_STRUCTURED_MAX_INTRINSIC_DIM +
      DROWSE_STRUCTURED_MAX_CURVE_NODES * DROWSE_STRUCTURED_MAX_RANK)
  );
}

export function decodeDrowseStructuredHookProfile(
  values: ArrayLike<number>,
  expectedLayerCount: number,
  expectedHiddenSize: number,
): DrowseStructuredHookProfile {
  if (values.length !== DROWSE_STRUCTURED_PROFILE_DESCRIPTOR_LENGTH) {
    throw new TypeError("Drowse structured hook profile has the wrong length");
  }
  const descriptor = Array.from(values);
  if (descriptor.some((value) => !Number.isSafeInteger(value))) {
    throw new TypeError("Drowse structured hook profile must contain integers");
  }
  const expected = [
    DROWSE_STRUCTURED_PROFILE_MAGIC,
    DROWSE_STRUCTURED_PROFILE_SCHEMA_VERSION,
    3,
    4,
    3,
    expectedLayerCount,
    expectedHiddenSize,
    DROWSE_STRUCTURED_MAX_AFFINE_GROUPS,
    DROWSE_STRUCTURED_MAX_RANK,
    DROWSE_STRUCTURED_MAX_PROBES,
    DROWSE_STRUCTURED_MAX_CURVES,
    DROWSE_STRUCTURED_MAX_CURVE_NODES,
    DROWSE_STRUCTURED_MAX_INTRINSIC_DIM,
    DROWSE_STRUCTURED_MAX_EMBED_DIM,
    DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE,
    DROWSE_STRUCTURED_MIN_COMPUTE_WORKGROUP_STORAGE_SIZE,
    DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES,
    DROWSE_STRUCTURED_MAX_WHITENER_RANK,
    DROWSE_STRUCTURED_MAX_GEOMETRY_CANDIDATES,
    DROWSE_STRUCTURED_GEOMETRY_OUTPUT_STRIDE,
    DROWSE_STRUCTURED_GEOMETRY_FOOT_RESTARTS,
    DROWSE_STRUCTURED_GEOMETRY_FOOT_ITERATIONS,
    DROWSE_STRUCTURED_GEOMETRY_WARM_RESTARTS,
    DROWSE_STRUCTURED_GEOMETRY_WARM_ITERATIONS,
    DROWSE_STRUCTURED_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
    DROWSE_STRUCTURED_GEOMETRY_KERNEL_STORAGE_BINDINGS,
    DROWSE_STRUCTURED_GEOMETRY_HEADER_STRIDE,
    drowseGeometryPayloadElements(expectedLayerCount, expectedHiddenSize),
    DROWSE_EXACT_READOUT_ABI_VERSION,
    DROWSE_READOUT_TOP_K,
    DROWSE_EXACT_READOUT_MAX_SAE_FEATURES_PER_CHUNK,
  ];
  if (descriptor.some((value, index) => value !== expected[index])) {
    throw new TypeError(
      "Drowse structured hook profile does not match the compiled model",
    );
  }
  return {
    schemaVersion: DROWSE_STRUCTURED_PROFILE_SCHEMA_VERSION,
    id: DROWSE_STRUCTURED_PROFILE_ID,
    hookAbi: DROWSE_HOOK_ABI,
    format: DROWSE_STRUCTURED_HOOK_FORMAT,
    layerCount: descriptor[5],
    hiddenSize: descriptor[6],
    maxAffineGroups: descriptor[7],
    maxRank: descriptor[8],
    maxProbes: descriptor[9],
    maxCurves: descriptor[10],
    maxCurveNodes: descriptor[11],
    maxIntrinsicDim: descriptor[12],
    maxEmbedDim: descriptor[13],
    curveParameterStride: descriptor[14],
    maxComputeWorkgroupStorageSize: descriptor[15],
    maxGeometryProbes: descriptor[16],
    maxWhitenerRank: descriptor[17],
    maxGeometryCandidates: descriptor[18],
    geometryOutputStride: descriptor[19],
    geometryFootRestarts: descriptor[20],
    geometryFootIterations: descriptor[21],
    geometryWarmRestarts: descriptor[22],
    geometryWarmIterations: descriptor[23],
    requiredMaxStorageBuffersPerShaderStage: descriptor[24],
    geometryKernelStorageBindings: descriptor[25],
    geometryHeaderStride: descriptor[26],
    geometryPayloadElements: descriptor[27],
    exactReadoutAbiVersion: DROWSE_EXACT_READOUT_ABI_VERSION,
    exactReadoutAbi: DROWSE_EXACT_READOUT_ABI,
    readoutTopK: descriptor[29],
    maxSaeFeaturesPerChunk: descriptor[30],
  };
}

const CURVE_PARAMETER_OFFSETS = {
  active: 0,
  intrinsicDim: 1,
  nodeParameters: 2,
  rbfWeights: 258,
  polynomial: 514,
  coordinateOffset: 586,
  coordinateScale: 594,
  origin: 602,
  target: 606,
  along: 610,
  onto: 611,
  bounds: 612,
  axisPeriodic: 620,
  axisPeriod: 624,
  sigmaPresent: 628,
  sigmaRbfWeights: 629,
  sigmaPolynomial: 661,
  damping: 670,
} as const;

export interface DrowseStructuredProgram {
  hookAbi: typeof DROWSE_HOOK_ABI;
  format: DrowseStructuredHookFormat;
  hiddenSize: number;
  layerCount: number;
  affineActive: Uint32Array;
  affineBasis: Float32Array;
  affineNeutral: Float32Array;
  affineTarget: Float32Array;
  affineAlong: Float32Array;
  affineKappa: Float32Array;
  probeKind: Uint32Array;
  probeDirection: Float32Array;
  probeBias: Float32Array;
  probeThreshold: Float32Array;
  curveActive: Uint32Array;
  curveRank: Uint32Array;
  curveIntrinsicDim: Uint32Array;
  curveEmbedDim: Uint32Array;
  curveNodeCount: Uint32Array;
  curveBasis: Float32Array;
  curveNeutral: Float32Array;
  curveNodeParameters: Float32Array;
  curveRbfWeights: Float32Array;
  curvePolynomial: Float32Array;
  curveCoordinateOffset: Float32Array;
  curveCoordinateScale: Float32Array;
  curveOrigin: Float32Array;
  curveTarget: Float32Array;
  curveAlong: Float32Array;
  curveOnto: Float32Array;
  curveBounds: Float32Array;
  curveAxisPeriodic: Uint32Array;
  curveAxisPeriod: Float32Array;
  curveSigmaPresent: Uint32Array;
  curveSigmaRbfWeights: Float32Array;
  curveSigmaPolynomial: Float32Array;
  curveDamping: Float32Array;
  curveDomainKind?: Uint32Array;
  whitenerRank?: Uint32Array;
  whitenerRidge?: Float32Array;
  whitenerBasis?: Float32Array;
  whitenerCorrection?: Float32Array;
  geometryActive?: Uint32Array;
  geometryKind?: Uint32Array;
  geometryRank?: Uint32Array;
  geometryIntrinsicDim?: Uint32Array;
  geometryCandidateCount?: Uint32Array;
  geometryCurveNodeCount?: Uint32Array;
  geometryDomainKind?: Uint32Array;
  geometryMean?: Float32Array;
  geometryInverseMean?: Float32Array;
  geometryBasis?: Float32Array;
  geometryGramInverse?: Float32Array;
  geometryCholesky?: Float32Array;
  geometryNodeWhite?: Float32Array;
  geometryCoordMap?: Float32Array;
  geometryCoordBias?: Float32Array;
  geometryCurveParameters?: Float32Array;
  geometryCurveNodeCoords?: Float32Array;
  geometryCurveNodeValues?: Float32Array;
  geometryFeet?: Float32Array;
  jLensBindingId?: string;
  jLensLayerIndices?: Int32Array;
  jLensTokenIds?: Int32Array;
  saeBindingId?: string;
}

export function packDrowseCurveParameters(
  program: DrowseStructuredProgram,
): Float32Array {
  const slots = program.layerCount * DROWSE_STRUCTURED_MAX_CURVES;
  const packed = new Float32Array(
    slots * DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE,
  );
  const copySlots = (
    values: Float32Array | Uint32Array,
    width: number,
    offset: number,
  ): void => {
    for (let slot = 0; slot < slots; slot += 1) {
      const sourceOffset = slot * width;
      packed.set(
        values.subarray(sourceOffset, sourceOffset + width),
        slot * DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE + offset,
      );
    }
  };
  copySlots(program.curveActive, 1, CURVE_PARAMETER_OFFSETS.active);
  copySlots(program.curveIntrinsicDim, 1, CURVE_PARAMETER_OFFSETS.intrinsicDim);
  copySlots(
    program.curveNodeParameters,
    DROWSE_STRUCTURED_MAX_CURVE_NODES * DROWSE_STRUCTURED_MAX_EMBED_DIM,
    CURVE_PARAMETER_OFFSETS.nodeParameters,
  );
  copySlots(
    program.curveRbfWeights,
    DROWSE_STRUCTURED_MAX_CURVE_NODES * DROWSE_STRUCTURED_MAX_RANK,
    CURVE_PARAMETER_OFFSETS.rbfWeights,
  );
  copySlots(
    program.curvePolynomial,
    (DROWSE_STRUCTURED_MAX_EMBED_DIM + 1) * DROWSE_STRUCTURED_MAX_RANK,
    CURVE_PARAMETER_OFFSETS.polynomial,
  );
  copySlots(
    program.curveCoordinateOffset,
    DROWSE_STRUCTURED_MAX_EMBED_DIM,
    CURVE_PARAMETER_OFFSETS.coordinateOffset,
  );
  copySlots(
    program.curveCoordinateScale,
    DROWSE_STRUCTURED_MAX_EMBED_DIM,
    CURVE_PARAMETER_OFFSETS.coordinateScale,
  );
  copySlots(
    program.curveOrigin,
    DROWSE_STRUCTURED_MAX_INTRINSIC_DIM,
    CURVE_PARAMETER_OFFSETS.origin,
  );
  copySlots(
    program.curveTarget,
    DROWSE_STRUCTURED_MAX_INTRINSIC_DIM,
    CURVE_PARAMETER_OFFSETS.target,
  );
  copySlots(program.curveAlong, 1, CURVE_PARAMETER_OFFSETS.along);
  copySlots(program.curveOnto, 1, CURVE_PARAMETER_OFFSETS.onto);
  copySlots(
    program.curveBounds,
    DROWSE_STRUCTURED_MAX_INTRINSIC_DIM * 2,
    CURVE_PARAMETER_OFFSETS.bounds,
  );
  copySlots(
    program.curveAxisPeriodic,
    DROWSE_STRUCTURED_MAX_INTRINSIC_DIM,
    CURVE_PARAMETER_OFFSETS.axisPeriodic,
  );
  copySlots(
    program.curveAxisPeriod,
    DROWSE_STRUCTURED_MAX_INTRINSIC_DIM,
    CURVE_PARAMETER_OFFSETS.axisPeriod,
  );
  copySlots(program.curveSigmaPresent, 1, CURVE_PARAMETER_OFFSETS.sigmaPresent);
  copySlots(
    program.curveSigmaRbfWeights,
    DROWSE_STRUCTURED_MAX_CURVE_NODES,
    CURVE_PARAMETER_OFFSETS.sigmaRbfWeights,
  );
  copySlots(
    program.curveSigmaPolynomial,
    DROWSE_STRUCTURED_MAX_EMBED_DIM + 1,
    CURVE_PARAMETER_OFFSETS.sigmaPolynomial,
  );
  copySlots(program.curveDamping, 1, CURVE_PARAMETER_OFFSETS.damping);
  return packed;
}

export function updatePackedDrowseCurveActive(
  packed: Float32Array,
  active: Uint32Array,
): void {
  if (
    packed.length !==
    active.length * DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE
  ) {
    throw new TypeError(
      "Drowse curve controls do not match the packed program",
    );
  }
  for (let slot = 0; slot < active.length; slot += 1) {
    packed[slot * DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE] = active[slot];
  }
}

export function packDrowseGeometryHeader(
  program: DrowseStructuredProgram,
): Uint32Array {
  const slots = program.layerCount * DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES;
  const packed = new Uint32Array(
    slots * DROWSE_STRUCTURED_GEOMETRY_HEADER_STRIDE,
  );
  const fields = [
    program.geometryActive!,
    program.geometryKind!,
    program.geometryRank!,
    program.geometryIntrinsicDim!,
    program.geometryCandidateCount!,
    program.geometryCurveNodeCount!,
    program.geometryDomainKind!,
  ];
  if (fields.length !== DROWSE_STRUCTURED_GEOMETRY_HEADER_STRIDE) {
    throw new TypeError("Drowse geometry header layout is inconsistent");
  }
  for (let slot = 0; slot < slots; slot += 1) {
    for (let field = 0; field < fields.length; field += 1) {
      packed[slot * DROWSE_STRUCTURED_GEOMETRY_HEADER_STRIDE + field] =
        fields[field][slot];
    }
  }
  return packed;
}

export function packDrowseGeometryPayload(
  program: DrowseStructuredProgram,
): Float32Array {
  const packed = new Float32Array(
    drowseGeometryPayloadElements(program.layerCount, program.hiddenSize),
  );
  const fields = [
    program.geometryMean!,
    program.geometryInverseMean!,
    program.geometryBasis!,
    program.geometryGramInverse!,
    program.geometryCholesky!,
    program.geometryNodeWhite!,
    program.geometryCoordMap!,
    program.geometryCoordBias!,
    program.geometryCurveParameters!,
    program.geometryCurveNodeCoords!,
    program.geometryCurveNodeValues!,
  ];
  let offset = 0;
  for (const field of fields) {
    packed.set(field, offset);
    offset += field.length;
  }
  if (offset !== packed.length || packed.byteLength % 4 !== 0) {
    throw new TypeError("Drowse geometry payload layout is inconsistent");
  }
  return packed;
}

export interface DrowseResidualCapture {
  layerCount: number;
  positionCount: number;
  hiddenSize: number;
  positions: number[];
  values: Float32Array;
}

export interface DrowseRankOneResidualCaptureV1 extends DrowseResidualCapture {
  abiVersion: 1;
  measurements: Float32Array;
}

export interface DrowseCaptureMessage {
  role: "system" | "user" | "assistant";
  content: string;
  roleName?: string;
}

export interface DrowseCaptureRow {
  system: string;
  messages: DrowseCaptureMessage[];
}

export interface DrowsePreparedCaptureRow {
  inputIds: number[];
  position: number;
}

export function validateDrowseCaptureRows(
  rows: DrowseCaptureRow[],
  specialTokenIds: number[],
): void {
  if (rows.length === 0) {
    throw new TypeError("Drowse residual capture requires at least one row");
  }
  for (const row of rows) {
    if (typeof row.system !== "string" || row.messages.length === 0) {
      throw new TypeError("Each Drowse capture row requires messages");
    }
    for (const message of row.messages) {
      if (
        (message.role !== "system" &&
          message.role !== "user" &&
          message.role !== "assistant") ||
        typeof message.content !== "string" ||
        (message.roleName !== undefined &&
          (typeof message.roleName !== "string" || !message.roleName.trim()))
      ) {
        throw new TypeError("A Drowse capture row contains an invalid message");
      }
      if (message.role === "system" && message.roleName !== undefined) {
        throw new TypeError(
          "Drowse system capture messages cannot name a role",
        );
      }
    }
    if (row.messages.at(-1)?.role !== "assistant") {
      throw new TypeError(
        "A Drowse capture row must end with an assistant message",
      );
    }
  }
  let previous = -1;
  for (const tokenId of specialTokenIds) {
    if (!Number.isSafeInteger(tokenId) || tokenId < 0 || tokenId <= previous) {
      throw new TypeError(
        "Drowse special token IDs must be sorted unique non-negative integers",
      );
    }
    previous = tokenId;
  }
}

export function validateDrowseCapturePositions(
  positions: number[],
  tokenCount: number,
): void {
  if (!Number.isSafeInteger(tokenCount) || tokenCount <= 0) {
    throw new TypeError("Drowse residual capture requires at least one token");
  }
  if (positions.length === 0) {
    throw new TypeError(
      "Drowse residual capture requires at least one position",
    );
  }
  let previous = -1;
  for (const position of positions) {
    if (
      !Number.isSafeInteger(position) ||
      position <= previous ||
      position >= tokenCount
    ) {
      throw new TypeError(
        `Drowse capture positions must be strictly increasing integers in [0, ${tokenCount})`,
      );
    }
    previous = position;
  }
}

export function validateDrowseRankOneProgram(
  program: DrowseRankOneProgram,
  expectedHiddenSize: number,
  expectedLayerCount: number,
): void {
  if (program.hookAbi !== DROWSE_HOOK_ABI) {
    throw new TypeError(
      `Unsupported Drowse hook ABI ${String(program.hookAbi)}`,
    );
  }
  if (
    program.hiddenSize !== expectedHiddenSize ||
    program.layerCount !== expectedLayerCount
  ) {
    throw new TypeError(
      `Drowse hook program targets ${program.layerCount}x${program.hiddenSize}; ` +
        `the model requires ${expectedLayerCount}x${expectedHiddenSize}`,
    );
  }
  const vectorLength = expectedHiddenSize * expectedLayerCount;
  requireTypedLength(
    program.enabled,
    Uint32Array,
    expectedLayerCount,
    "enabled",
  );
  requireTypedLength(program.basis, Float32Array, vectorLength, "basis");
  requireTypedLength(program.neutral, Float32Array, vectorLength, "neutral");
  requireTypedLength(
    program.target,
    Float32Array,
    expectedLayerCount,
    "target",
  );
  requireTypedLength(program.along, Float32Array, expectedLayerCount, "along");
  requireTypedLength(
    program.collapse,
    Float32Array,
    expectedLayerCount,
    "collapse",
  );
  requireTypedLength(
    program.probeBasis,
    Float32Array,
    vectorLength,
    "probeBasis",
  );
  requireTypedLength(
    program.probeNeutral,
    Float32Array,
    vectorLength,
    "probeNeutral",
  );

  for (const value of program.enabled) {
    if (value !== 0 && value !== 1) {
      throw new TypeError("Drowse enabled values must be zero or one");
    }
  }
  for (const [label, values] of [
    ["basis", program.basis],
    ["neutral", program.neutral],
    ["target", program.target],
    ["along", program.along],
    ["collapse", program.collapse],
    ["probeBasis", program.probeBasis],
    ["probeNeutral", program.probeNeutral],
  ] as const) {
    for (const value of values) {
      if (!Number.isFinite(value)) {
        throw new TypeError(`Drowse ${label} values must be finite`);
      }
    }
  }
  for (let layer = 0; layer < expectedLayerCount; layer += 1) {
    if (program.enabled[layer] === 1) {
      requireUnitLayer(program.basis, layer, expectedHiddenSize, "basis");
    }
    requireUnitLayer(
      program.probeBasis,
      layer,
      expectedHiddenSize,
      "probeBasis",
      true,
    );
  }
}

export function validateDrowseSaeDictionary(
  dictionary: DrowseSaeDictionary,
  hiddenSize: number,
  layerCount: number,
): void {
  if (dictionary === null || typeof dictionary !== "object") {
    throw new TypeError("Drowse SAE dictionary is invalid");
  }
  if (
    dictionary.hookAbi !== DROWSE_HOOK_ABI ||
    !validDrowseBindingId(dictionary.bindingId) ||
    dictionary.hiddenSize !== hiddenSize
  ) {
    throw new TypeError(
      `Drowse SAE dictionary targets ${String(dictionary.hiddenSize)} hidden dimensions under ${String(dictionary.hookAbi)}`,
    );
  }
  if (
    !Number.isSafeInteger(dictionary.runtimeLayerIndex) ||
    dictionary.runtimeLayerIndex < 0 ||
    dictionary.runtimeLayerIndex >= layerCount
  ) {
    throw new TypeError("Drowse SAE layer is invalid");
  }
  if (
    !Number.isSafeInteger(dictionary.featureCount) ||
    dictionary.featureCount < 1 ||
    dictionary.featureCount > 0x7fffffff
  ) {
    throw new TypeError("Drowse SAE feature count is invalid");
  }
  const encoderElements = hiddenSize * dictionary.featureCount;
  if (!Number.isSafeInteger(encoderElements)) {
    throw new TypeError("Drowse SAE encoder shape is invalid");
  }
  requireTypedLength(
    dictionary.encoder,
    Float32Array,
    encoderElements,
    "SAE encoder",
  );
  requireTypedLength(
    dictionary.encoderBias,
    Float32Array,
    dictionary.featureCount,
    "SAE encoder bias",
  );
  requireTypedLength(
    dictionary.encoderThreshold,
    Float32Array,
    dictionary.featureCount,
    "SAE encoder threshold",
  );
  requireTypedLength(
    dictionary.decoderBias,
    Float32Array,
    hiddenSize,
    "SAE decoder bias",
  );
  if (!(["relu", "jump_relu"] as const).includes(dictionary.activation)) {
    throw new TypeError("Drowse SAE activation is invalid");
  }
  if (
    dictionary.activation === "relu" &&
    dictionary.encoderThreshold.some((value) => value !== 0)
  ) {
    throw new TypeError("A ReLU Drowse SAE must use zero encoder thresholds");
  }
  for (const [label, values] of [
    ["encoder", dictionary.encoder],
    ["encoder bias", dictionary.encoderBias],
    ["encoder threshold", dictionary.encoderThreshold],
    ["decoder bias", dictionary.decoderBias],
  ] as const) {
    if (values.some((value) => !Number.isFinite(value))) {
      throw new TypeError(`Drowse SAE ${label} values must be finite`);
    }
  }
}

export function validateDrowseJlensDictionary(
  dictionary: DrowseJlensDictionary,
  hiddenSize: number,
  layerCount: number,
): void {
  if (dictionary === null || typeof dictionary !== "object") {
    throw new TypeError("Drowse J-lens dictionary is invalid");
  }
  if (
    dictionary.hookAbi !== DROWSE_HOOK_ABI ||
    !validDrowseBindingId(dictionary.bindingId) ||
    dictionary.hiddenSize !== hiddenSize ||
    !(dictionary.layerIndices instanceof Int32Array) ||
    !Array.isArray(dictionary.matrices)
  ) {
    throw new TypeError("Drowse J-lens dictionary identity is invalid");
  }
  if (
    dictionary.layerIndices.length < 1 ||
    dictionary.layerIndices.some(
      (layer, index) =>
        layer < 0 ||
        layer >= layerCount ||
        (index > 0 && layer <= dictionary.layerIndices[index - 1]),
    )
  ) {
    throw new TypeError(
      "Drowse J-lens dictionary layers must be unique, ordered, and in range",
    );
  }
  if (
    dictionary.matrices.length !== dictionary.layerIndices.length ||
    dictionary.matrices.some(
      (matrix) =>
        !(matrix instanceof Float32Array) ||
        matrix.length !== hiddenSize * hiddenSize,
    )
  ) {
    throw new TypeError("Drowse J-lens Jacobians have the wrong shape");
  }
}

export function validateDrowseStructuredProgram(
  program: DrowseStructuredProgram,
  expectedHiddenSize: number,
  expectedLayerCount: number,
): void {
  if (
    program.hookAbi !== DROWSE_HOOK_ABI ||
    (program.format !== DROWSE_STRUCTURED_HOOK_FORMAT &&
      program.format !== DROWSE_STRUCTURED_HOOK_FORMAT_V2)
  ) {
    throw new TypeError("Unsupported Drowse structured hook program identity");
  }
  if (
    program.hiddenSize !== expectedHiddenSize ||
    program.layerCount !== expectedLayerCount
  ) {
    throw new TypeError(
      `Drowse hook program targets ${program.layerCount}x${program.hiddenSize}; ` +
        `the model requires ${expectedLayerCount}x${expectedHiddenSize}`,
    );
  }
  const l = expectedLayerCount;
  const d = expectedHiddenSize;
  const g = DROWSE_STRUCTURED_MAX_AFFINE_GROUPS;
  const r = DROWSE_STRUCTURED_MAX_RANK;
  const p = DROWSE_STRUCTURED_MAX_PROBES;
  const n = DROWSE_STRUCTURED_MAX_CURVE_NODES;
  const c = DROWSE_STRUCTURED_MAX_CURVES;
  const i = DROWSE_STRUCTURED_MAX_INTRINSIC_DIM;
  const m = DROWSE_STRUCTURED_MAX_EMBED_DIM;
  const curveSlots = l * c;
  requireTypedLength(program.affineActive, Uint32Array, l * g, "affineActive");
  requireTypedLength(
    program.affineBasis,
    Float32Array,
    l * g * r * d,
    "affineBasis",
  );
  requireTypedLength(
    program.affineNeutral,
    Float32Array,
    l * g * d,
    "affineNeutral",
  );
  requireTypedLength(
    program.affineTarget,
    Float32Array,
    l * g * r,
    "affineTarget",
  );
  requireTypedLength(program.affineAlong, Float32Array, l * g, "affineAlong");
  requireTypedLength(
    program.affineKappa,
    Float32Array,
    l * g * r,
    "affineKappa",
  );
  requireTypedLength(program.probeKind, Uint32Array, l * p, "probeKind");
  requireTypedLength(
    program.probeDirection,
    Float32Array,
    l * p * d,
    "probeDirection",
  );
  requireTypedLength(program.probeBias, Float32Array, l * p, "probeBias");
  requireTypedLength(
    program.probeThreshold,
    Float32Array,
    l * p,
    "probeThreshold",
  );
  requireTypedLength(
    program.curveActive,
    Uint32Array,
    curveSlots,
    "curveActive",
  );
  requireTypedLength(program.curveRank, Uint32Array, curveSlots, "curveRank");
  requireTypedLength(
    program.curveIntrinsicDim,
    Uint32Array,
    curveSlots,
    "curveIntrinsicDim",
  );
  requireTypedLength(
    program.curveEmbedDim,
    Uint32Array,
    curveSlots,
    "curveEmbedDim",
  );
  requireTypedLength(
    program.curveNodeCount,
    Uint32Array,
    curveSlots,
    "curveNodeCount",
  );
  requireTypedLength(
    program.curveBasis,
    Float32Array,
    curveSlots * r * d,
    "curveBasis",
  );
  requireTypedLength(
    program.curveNeutral,
    Float32Array,
    curveSlots * d,
    "curveNeutral",
  );
  requireTypedLength(
    program.curveNodeParameters,
    Float32Array,
    curveSlots * n * m,
    "curveNodeParameters",
  );
  requireTypedLength(
    program.curveRbfWeights,
    Float32Array,
    curveSlots * n * r,
    "curveRbfWeights",
  );
  requireTypedLength(
    program.curvePolynomial,
    Float32Array,
    curveSlots * (m + 1) * r,
    "curvePolynomial",
  );
  for (const [label, values] of [
    ["curveAlong", program.curveAlong],
    ["curveOnto", program.curveOnto],
    ["curveDamping", program.curveDamping],
  ] as const) {
    requireTypedLength(values, Float32Array, curveSlots, label);
  }
  requireTypedLength(
    program.curveCoordinateOffset,
    Float32Array,
    curveSlots * m,
    "curveCoordinateOffset",
  );
  requireTypedLength(
    program.curveCoordinateScale,
    Float32Array,
    curveSlots * m,
    "curveCoordinateScale",
  );
  requireTypedLength(
    program.curveOrigin,
    Float32Array,
    curveSlots * i,
    "curveOrigin",
  );
  requireTypedLength(
    program.curveTarget,
    Float32Array,
    curveSlots * i,
    "curveTarget",
  );
  requireTypedLength(
    program.curveBounds,
    Float32Array,
    curveSlots * i * 2,
    "curveBounds",
  );
  requireTypedLength(
    program.curveAxisPeriodic,
    Uint32Array,
    curveSlots * i,
    "curveAxisPeriodic",
  );
  requireTypedLength(
    program.curveAxisPeriod,
    Float32Array,
    curveSlots * i,
    "curveAxisPeriod",
  );
  requireTypedLength(
    program.curveSigmaPresent,
    Uint32Array,
    curveSlots,
    "curveSigmaPresent",
  );
  requireTypedLength(
    program.curveSigmaRbfWeights,
    Float32Array,
    curveSlots * n,
    "curveSigmaRbfWeights",
  );
  requireTypedLength(
    program.curveSigmaPolynomial,
    Float32Array,
    curveSlots * (m + 1),
    "curveSigmaPolynomial",
  );
  for (const [label, values] of Object.entries(program)) {
    if (!(values instanceof Float32Array)) continue;
    for (const value of values) {
      if (!Number.isFinite(value))
        throw new TypeError(`Drowse ${label} values must be finite`);
    }
  }
  for (const values of [
    program.affineActive,
    program.curveActive,
    program.curveSigmaPresent,
    program.curveAxisPeriodic,
  ]) {
    for (const value of values) {
      if (value !== 0 && value !== 1)
        throw new TypeError("Drowse active values must be zero or one");
    }
  }
  for (const value of program.probeKind) {
    if (value > 4) throw new TypeError("Drowse probe kind is invalid");
  }
  const hasJlensProbe = program.probeKind.some((value) => value === 3);
  const hasJlensProgram = program.jLensBindingId !== undefined;
  if (
    hasJlensProgram !== (program.jLensLayerIndices !== undefined) ||
    hasJlensProgram !== (program.jLensTokenIds !== undefined) ||
    (hasJlensProbe && !hasJlensProgram)
  ) {
    throw new TypeError(
      "Drowse J-lens probes require one complete probability program",
    );
  }
  if (hasJlensProgram) {
    if (!validDrowseBindingId(program.jLensBindingId)) {
      throw new TypeError("Drowse J-lens binding is invalid");
    }
    const activeLayers: number[] = [];
    for (let layer = 0; layer < l; layer += 1) {
      const start = layer * p;
      if (
        program.probeKind
          .subarray(start, start + p)
          .some((value) => value === 3)
      ) {
        activeLayers.push(layer);
      }
    }
    if (
      program.jLensLayerIndices!.some(
        (layer, index) =>
          layer < 0 ||
          layer >= l ||
          (index > 0 && layer <= program.jLensLayerIndices![index - 1]),
      )
    ) {
      throw new TypeError(
        "Drowse J-lens layer indices must be unique, ordered, and in range",
      );
    }
    if (hasJlensProbe) {
      requireTypedLength(
        program.jLensLayerIndices!,
        Int32Array,
        activeLayers.length,
        "jLensLayerIndices",
      );
      for (let index = 0; index < activeLayers.length; index += 1) {
        if (program.jLensLayerIndices![index] !== activeLayers[index]) {
          throw new TypeError(
            "Drowse J-lens layer indices must exactly match the active probability layers",
          );
        }
      }
    }
    requireTypedLength(program.jLensTokenIds, Int32Array, p, "jLensTokenIds");
    for (const tokenId of program.jLensTokenIds!) {
      if (tokenId < 0)
        throw new TypeError("Drowse J-lens token IDs are invalid");
    }
  }
  if (
    program.saeBindingId !== undefined &&
    !validDrowseBindingId(program.saeBindingId)
  ) {
    throw new TypeError("Drowse SAE readout binding is invalid");
  }
  for (let layer = 0; layer < curveSlots; layer += 1) {
    if (
      program.curveRank[layer] > r ||
      program.curveIntrinsicDim[layer] > i ||
      program.curveEmbedDim[layer] > m ||
      program.curveNodeCount[layer] > n ||
      (program.curveActive[layer] === 1 &&
        (program.curveRank[layer] === 0 ||
          program.curveIntrinsicDim[layer] === 0 ||
          program.curveEmbedDim[layer] === 0 ||
          program.curveNodeCount[layer] < 2))
    ) {
      throw new TypeError(
        `Drowse curved layer ${layer} exceeds the supported shape`,
      );
    }
  }
  const geometryFields = [
    "curveDomainKind",
    "whitenerRank",
    "whitenerRidge",
    "whitenerBasis",
    "whitenerCorrection",
    "geometryActive",
    "geometryKind",
    "geometryRank",
    "geometryIntrinsicDim",
    "geometryCandidateCount",
    "geometryCurveNodeCount",
    "geometryDomainKind",
    "geometryMean",
    "geometryInverseMean",
    "geometryBasis",
    "geometryGramInverse",
    "geometryCholesky",
    "geometryNodeWhite",
    "geometryCoordMap",
    "geometryCoordBias",
    "geometryCurveParameters",
    "geometryCurveNodeCoords",
    "geometryCurveNodeValues",
    "geometryFeet",
  ] as const;
  if (program.format === DROWSE_STRUCTURED_HOOK_FORMAT_V2) {
    if (geometryFields.some((field) => program[field] !== undefined)) {
      throw new TypeError(
        "Drowse structured-v2 programs cannot contain geometry buffers",
      );
    }
    return;
  }
  for (const field of geometryFields) {
    if (program[field] === undefined) {
      throw new TypeError(`Drowse structured-v3 program is missing ${field}`);
    }
  }
  const geometrySlots = l * DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES;
  const geometryRankSlots = geometrySlots * r;
  requireTypedLength(program.whitenerRank, Uint32Array, l, "whitenerRank");
  requireTypedLength(program.whitenerRidge, Float32Array, l, "whitenerRidge");
  requireTypedLength(
    program.whitenerBasis,
    Float32Array,
    l * DROWSE_STRUCTURED_MAX_WHITENER_RANK * d,
    "whitenerBasis",
  );
  requireTypedLength(
    program.whitenerCorrection,
    Float32Array,
    l * DROWSE_STRUCTURED_MAX_WHITENER_RANK,
    "whitenerCorrection",
  );
  for (const field of [
    "geometryActive",
    "geometryKind",
    "geometryRank",
    "geometryIntrinsicDim",
    "geometryCandidateCount",
    "geometryCurveNodeCount",
    "geometryDomainKind",
  ] as const) {
    requireTypedLength(program[field], Uint32Array, geometrySlots, field);
  }
  requireTypedLength(
    program.curveDomainKind,
    Uint32Array,
    curveSlots,
    "curveDomainKind",
  );
  requireTypedLength(
    program.geometryMean,
    Float32Array,
    geometrySlots * d,
    "geometryMean",
  );
  requireTypedLength(
    program.geometryInverseMean,
    Float32Array,
    geometrySlots * d,
    "geometryInverseMean",
  );
  requireTypedLength(
    program.geometryBasis,
    Float32Array,
    geometryRankSlots * d,
    "geometryBasis",
  );
  requireTypedLength(
    program.geometryGramInverse,
    Float32Array,
    geometrySlots * r * r,
    "geometryGramInverse",
  );
  requireTypedLength(
    program.geometryCholesky,
    Float32Array,
    geometrySlots * r * r,
    "geometryCholesky",
  );
  requireTypedLength(
    program.geometryNodeWhite,
    Float32Array,
    geometryRankSlots * DROWSE_STRUCTURED_MAX_GEOMETRY_CANDIDATES,
    "geometryNodeWhite",
  );
  requireTypedLength(
    program.geometryCoordMap,
    Float32Array,
    geometrySlots * i * r,
    "geometryCoordMap",
  );
  requireTypedLength(
    program.geometryCoordBias,
    Float32Array,
    geometrySlots * i,
    "geometryCoordBias",
  );
  requireTypedLength(
    program.geometryCurveParameters,
    Float32Array,
    geometrySlots * DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE,
    "geometryCurveParameters",
  );
  requireTypedLength(
    program.geometryCurveNodeCoords,
    Float32Array,
    geometrySlots * n * i,
    "geometryCurveNodeCoords",
  );
  requireTypedLength(
    program.geometryCurveNodeValues,
    Float32Array,
    geometrySlots * n * r,
    "geometryCurveNodeValues",
  );
  requireTypedLength(
    program.geometryFeet,
    Float32Array,
    geometrySlots * i,
    "geometryFeet",
  );
  for (let layer = 0; layer < l; layer += 1) {
    if (program.whitenerRank![layer] > DROWSE_STRUCTURED_MAX_WHITENER_RANK) {
      throw new TypeError(
        `Drowse whitener layer ${layer} exceeds the supported rank`,
      );
    }
    const layerActive = program
      .geometryActive!.subarray(
        layer * DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES,
        (layer + 1) * DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES,
      )
      .some((value) => value === 1);
    if (
      layerActive &&
      (program.whitenerRank![layer] === 0 ||
        !(program.whitenerRidge![layer] > 0))
    ) {
      throw new TypeError(
        `Drowse active geometry layer ${layer} requires a positive-definite whitener`,
      );
    }
  }
  for (let slot = 0; slot < geometrySlots; slot += 1) {
    const active = program.geometryActive![slot];
    const kind = program.geometryKind![slot];
    const rank = program.geometryRank![slot];
    const intrinsicDim = program.geometryIntrinsicDim![slot];
    const candidates = program.geometryCandidateCount![slot];
    const curveNodes = program.geometryCurveNodeCount![slot];
    const domain = program.geometryDomainKind![slot];
    if (
      active > 1 ||
      kind > 2 ||
      domain > 2 ||
      rank > r ||
      intrinsicDim > i ||
      candidates > DROWSE_STRUCTURED_MAX_GEOMETRY_CANDIDATES ||
      curveNodes > n ||
      (active === 1 &&
        (kind === 0 || rank === 0 || intrinsicDim === 0 || candidates === 0)) ||
      (active === 1 && kind === 2 && (curveNodes < 2 || domain === 0))
    ) {
      throw new TypeError(
        `Drowse geometry slot ${slot} exceeds the supported shape`,
      );
    }
  }
  for (let slot = 0; slot < curveSlots; slot += 1) {
    const domain = program.curveDomainKind![slot];
    if (domain > 2 || (program.curveActive[slot] === 1 && domain === 0)) {
      throw new TypeError(`Drowse curve domain slot ${slot} is invalid`);
    }
  }
}

function requireTypedLength<
  T extends
    | Uint32ArrayConstructor
    | Int32ArrayConstructor
    | Float32ArrayConstructor,
>(value: unknown, constructor: T, expected: number, label: string): void {
  if (!(value instanceof constructor) || value.length !== expected) {
    throw new TypeError(
      `Drowse ${label} must contain exactly ${expected} values`,
    );
  }
}

function validDrowseBindingId(value: unknown): value is string {
  return typeof value === "string" && /^[a-f0-9]{64}$/u.test(value);
}

function requireUnitLayer(
  values: Float32Array,
  layer: number,
  hiddenSize: number,
  label: string,
  allowZero = false,
): void {
  let squaredNorm = 0;
  const start = layer * hiddenSize;
  for (let index = start; index < start + hiddenSize; index += 1) {
    squaredNorm += values[index] * values[index];
  }
  if (allowZero && squaredNorm === 0) return;
  if (Math.abs(squaredNorm - 1) > 1e-3) {
    throw new TypeError(`Drowse ${label} layer ${layer} must be unit length`);
  }
}
