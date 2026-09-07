import { jest, expect, test } from "@jest/globals";
import { LLMChatPipeline } from "../src/llm_chat";

type AnyObj = Record<string, any>;

function makeAvailability(overrides: Partial<AnyObj> = {}) {
  return {
    prefill: false,
    batch_prefill: false,
    decode: false,
    batch_decode: false,
    create_tir_paged_kv_cache: false,
    create_rnn_state: false,
    sample_with_top_p: false,
    argsort_probs: false,
    ...overrides,
  };
}

function makeFakeTensor(seqLen: number) {
  const tensor: AnyObj = {
    shape: [seqLen, 4],
  };
  tensor.view = jest.fn((shape: number[]) => {
    tensor.shape = shape;
    return tensor;
  });
  return tensor;
}

test("legacy compiled hook names alias to Drowse without replacing current kernels", () => {
  const applyAliases = (LLMChatPipeline as AnyObj).applyLegacyDrowseAliases;
  const legacyPrefill = jest.fn();
  const currentDecode = jest.fn();
  const legacyDecode = jest.fn();
  const registry: AnyObj = {
    saklas_structured_batch_prefill: legacyPrefill,
    drowse_structured_batch_decode: currentDecode,
    saklas_structured_batch_decode: legacyDecode,
    unrelated: jest.fn(),
  };

  applyAliases(registry);

  expect(registry.drowse_structured_batch_prefill).toBe(legacyPrefill);
  expect(registry.drowse_structured_batch_decode).toBe(currentDecode);
  expect(registry.drowse_unrelated).toBeUndefined();
});

test("parseKVStateKind defaults missing metadata to kv_cache", () => {
  const pipeline = Object.create(LLMChatPipeline.prototype) as AnyObj;
  expect(pipeline.parseKVStateKind(undefined)).toBe("kv_cache");
  expect(pipeline.parseKVStateKind(null)).toBe("kv_cache");
});

test("resolveModelABI selects single ABI for kv_cache when prefill/decode exist", () => {
  const resolve = (LLMChatPipeline as AnyObj).resolveModelABI;
  const abi = resolve(
    "kv_cache",
    makeAvailability({
      prefill: true,
      decode: true,
      create_tir_paged_kv_cache: true,
    }),
  );
  expect(abi.prefillABI).toBe("single");
  expect(abi.decodeABI).toBe("single");
  expect(abi.needsKVCache).toBe(true);
  expect(abi.needsRNNState).toBe(false);
});

test("resolveModelABI selects batch ABI for kv_cache when only batch kernels exist", () => {
  const resolve = (LLMChatPipeline as AnyObj).resolveModelABI;
  const abi = resolve(
    "kv_cache",
    makeAvailability({
      batch_prefill: true,
      batch_decode: true,
      create_tir_paged_kv_cache: true,
    }),
  );
  expect(abi.prefillABI).toBe("batch");
  expect(abi.decodeABI).toBe("batch");
  expect(abi.needsKVCache).toBe(true);
  expect(abi.needsRNNState).toBe(false);
});

test("resolveModelABI requires create_rnn_state for rnn_state models", () => {
  const resolve = (LLMChatPipeline as AnyObj).resolveModelABI;
  expect(() =>
    resolve(
      "rnn_state",
      makeAvailability({
        prefill: true,
        decode: true,
      }),
    ),
  ).toThrow(/create_rnn_state/);
});

test("resolveModelABI requires both state creators and batch kernels for hybrid models", () => {
  const resolve = (LLMChatPipeline as AnyObj).resolveModelABI;
  expect(() =>
    resolve(
      "hybrid",
      makeAvailability({
        batch_prefill: true,
        batch_decode: true,
        create_rnn_state: true,
      }),
    ),
  ).toThrow(/create_tir_paged_kv_cache/);

  const abi = resolve(
    "hybrid",
    makeAvailability({
      batch_prefill: true,
      batch_decode: true,
      create_rnn_state: true,
      create_tir_paged_kv_cache: true,
    }),
  );
  expect(abi.prefillABI).toBe("batch");
  expect(abi.decodeABI).toBe("batch");
  expect(abi.needsKVCache).toBe(true);
  expect(abi.needsRNNState).toBe(true);
});

test("resolveModelABI rejects kv_state_kind none in chat pipeline", () => {
  const resolve = (LLMChatPipeline as AnyObj).resolveModelABI;
  expect(() => resolve("none", makeAvailability())).toThrow(
    /kv_state_kind=`none`/,
  );
});

test("batch kv_cache invoke path does not require rnnState", () => {
  const pipeline = Object.create(LLMChatPipeline.prototype) as AnyObj;
  pipeline.resolvedModelABI = {
    kvStateKind: "kv_cache",
    prefillABI: "batch",
    decodeABI: "batch",
    prefillFunctionName: "batch_prefill",
    decodeFunctionName: "batch_decode",
    needsKVCache: true,
    needsRNNState: false,
  };
  pipeline.prefill = jest.fn(() => ({ get: jest.fn() }));
  pipeline.decoding = jest.fn(() => ({ get: jest.fn() }));
  pipeline.prefillLogitPositionHost = new Int32Array(1);
  pipeline.prefillLogitPositions = { copyFrom: jest.fn() };
  pipeline.kvCache = { kind: "kv" };
  pipeline.params = { kind: "params" };

  pipeline.invokePrefill({ kind: "emb" }, 6);
  expect(pipeline.prefill).toHaveBeenCalledWith(
    { kind: "emb" },
    pipeline.prefillLogitPositions,
    pipeline.kvCache,
    pipeline.params,
  );

  pipeline.invokeDecode({ kind: "emb" });
  expect(pipeline.decoding).toHaveBeenCalledWith(
    { kind: "emb" },
    pipeline.kvCache,
    pipeline.params,
  );
});

test("hybrid invoke path passes kv cache and rnn state in ABI order", () => {
  const pipeline = Object.create(LLMChatPipeline.prototype) as AnyObj;
  pipeline.resolvedModelABI = {
    kvStateKind: "hybrid",
    prefillABI: "batch",
    decodeABI: "batch",
    prefillFunctionName: "batch_prefill",
    decodeFunctionName: "batch_decode",
    needsKVCache: true,
    needsRNNState: true,
  };
  pipeline.prefill = jest.fn(() => ({ get: jest.fn() }));
  pipeline.decoding = jest.fn(() => ({ get: jest.fn() }));
  pipeline.prefillLogitPositionHost = new Int32Array(1);
  pipeline.prefillLogitPositions = { copyFrom: jest.fn() };
  pipeline.kvCache = { kind: "kv" };
  pipeline.rnnState = { kind: "rnn" };
  pipeline.params = { kind: "params" };

  pipeline.invokePrefill({ kind: "emb" }, 4);
  expect(pipeline.prefill).toHaveBeenCalledWith(
    { kind: "emb" },
    pipeline.prefillLogitPositions,
    pipeline.kvCache,
    pipeline.rnnState,
    pipeline.params,
  );

  pipeline.invokeDecode({ kind: "emb" });
  expect(pipeline.decoding).toHaveBeenCalledWith(
    { kind: "emb" },
    pipeline.kvCache,
    pipeline.rnnState,
    pipeline.params,
  );
});

test("embedAndForward begins and ends forward for all active states", async () => {
  const pipeline = Object.create(LLMChatPipeline.prototype) as AnyObj;
  const kvState = { id: "kv" };
  const rnnState = { id: "rnn" };
  const logits = { value: "logits" };

  pipeline.prefillChunkSize = 1024;
  pipeline.resolvedModelABI = {
    kvStateKind: "hybrid",
    prefillABI: "batch",
    decodeABI: "batch",
    prefillFunctionName: "batch_prefill",
    decodeFunctionName: "batch_decode",
    needsKVCache: true,
    needsRNNState: true,
  };
  pipeline.kvCache = kvState;
  pipeline.rnnState = rnnState;
  pipeline.params = { kind: "params" };
  pipeline.prefillLogitPositionHost = new Int32Array(1);
  pipeline.prefillLogitPositions = { copyFrom: jest.fn() };
  pipeline.filledKVCacheLength = 0;
  pipeline.tvm = {
    beginScope: jest.fn(),
    endScope: jest.fn(),
    makeShapeTuple: jest.fn((x: number[]) => x),
    concatEmbeddings: jest.fn(),
    detachFromCurrentScope: jest.fn((x: any) => x),
    attachToCurrentScope: jest.fn(),
  };
  pipeline.fKVCacheBeginForward = jest.fn();
  pipeline.fKVCacheEndForward = jest.fn();
  pipeline.getTokensEmbeddings = jest.fn(() => makeFakeTensor(1));
  pipeline.getImageEmbeddings = jest.fn();
  pipeline.decoding = jest.fn(() => ({ get: jest.fn(() => logits) }));

  const out = await pipeline.embedAndForward([[101]], 1);
  expect(out).toBe(logits);
  expect(pipeline.fKVCacheBeginForward).toHaveBeenNthCalledWith(
    1,
    kvState,
    [0],
    [1],
  );
  expect(pipeline.fKVCacheBeginForward).toHaveBeenNthCalledWith(
    2,
    rnnState,
    [0],
    [1],
  );
  expect(pipeline.fKVCacheEndForward).toHaveBeenNthCalledWith(1, rnnState);
  expect(pipeline.fKVCacheEndForward).toHaveBeenNthCalledWith(2, kvState);
});

test("rank-one forward does not read the structured hidden-state tuple slot", async () => {
  const pipeline = Object.create(LLMChatPipeline.prototype) as AnyObj;
  const kvState = { id: "kv" };
  const logits = { value: "logits" };
  const measurements = { value: "measurements" };
  const retValue = {
    get: jest.fn((index: number) => {
      if (index === 0) return logits;
      if (index === 2) return measurements;
      throw new Error(`Index ${index} out of bounds 3`);
    }),
  };

  pipeline.prefillChunkSize = 1024;
  pipeline.resolvedModelABI = {
    kvStateKind: "kv_cache",
    prefillABI: "batch",
    decodeABI: "batch",
    prefillFunctionName: "batch_prefill",
    decodeFunctionName: "batch_decode",
    needsKVCache: true,
    needsRNNState: false,
  };
  pipeline.kvCache = kvState;
  pipeline.params = { kind: "params" };
  pipeline.filledKVCacheLength = 0;
  pipeline.drowseProgram = {
    enabled: {},
    basis: {},
    neutral: {},
    target: {},
    along: {},
    collapse: {},
    probeBasis: {},
    probeNeutral: {},
  };
  pipeline.drowseDecoding = jest.fn(() => retValue);
  pipeline.tvm = {
    beginScope: jest.fn(),
    endScope: jest.fn(),
    makeShapeTuple: jest.fn((x: number[]) => x),
    concatEmbeddings: jest.fn(),
    detachFromCurrentScope: jest.fn((x: any) => x),
    attachToCurrentScope: jest.fn(),
  };
  pipeline.fKVCacheBeginForward = jest.fn();
  pipeline.fKVCacheEndForward = jest.fn();
  pipeline.getTokensEmbeddings = jest.fn(() => makeFakeTensor(1));
  pipeline.getImageEmbeddings = jest.fn();

  await expect(pipeline.embedAndForward([[101]], 1)).resolves.toBe(logits);
  expect(retValue.get).toHaveBeenCalledWith(2);
  expect(retValue.get).toHaveBeenCalledWith(0);
  expect(retValue.get).not.toHaveBeenCalledWith(3);
});
