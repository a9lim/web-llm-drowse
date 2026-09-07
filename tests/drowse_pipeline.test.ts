import { jest, test, expect } from "@jest/globals";

import {
  LLMChatPipeline,
  planDrowseJlensChunks,
  planDrowseSaeChunks,
  DROWSE_JLENS_MAX_CHUNK_BYTES,
  DROWSE_SAE_MAX_CHUNK_BYTES,
  DROWSE_SAE_MAX_FEATURES_PER_CHUNK,
} from "../src/llm_chat";
import { DROWSE_HOOK_ABI, DrowseRankOneProgram } from "../src/drowse";

const JLENS_BINDING = "b".repeat(64);
const SAE_BINDING = "a".repeat(64);

function program(): DrowseRankOneProgram {
  const basis = new Float32Array(8);
  basis[0] = 1;
  return {
    hookAbi: DROWSE_HOOK_ABI,
    hiddenSize: 4,
    layerCount: 2,
    enabled: Uint32Array.from([1, 0]),
    basis,
    neutral: new Float32Array(8),
    target: new Float32Array(2),
    along: new Float32Array(2),
    collapse: new Float32Array(2),
    probeBasis: new Float32Array(8),
    probeNeutral: new Float32Array(8),
  };
}

function tensor() {
  return {
    shape: [2],
    copyFrom: jest.fn().mockReturnThis(),
    dispose: jest.fn(),
    toArray: jest.fn(() => Float32Array.from([1, 2])),
  };
}

function pipeline(): any {
  const result = Object.create(LLMChatPipeline.prototype) as any;
  result["drowsePrefill"] = jest.fn();
  result["drowseDecoding"] = jest.fn();
  result["config"] = {
    model_config: {
      text_config: { hidden_size: 4, num_hidden_layers: 2 },
    },
  };
  result["device"] = { sync: jest.fn(async () => undefined) };
  result["drowseJlensBufferLimits"] = {
    maxBufferSize: 1 << 28,
    maxStorageBufferBindingSize: 1 << 27,
  };
  result["tvm"] = {
    beginScope: jest.fn(),
    endScope: jest.fn(),
    empty: jest.fn((shape: number[]) => ({ ...tensor(), shape })),
    detachFromCurrentScope: jest.fn((value: unknown) => value),
    cpu: jest.fn(() => "cpu"),
  };
  return result;
}

test("runtime capabilities expose scene stitching only for a supported user seat", () => {
  const result = pipeline();
  result["conversation"] = {
    supportsDrowseNamedRoles: jest.fn(() => false),
    supportsDrowseUserSeatGeneration: jest.fn(() => true),
  };

  expect(result.getDrowseRuntimeCapabilities()).toEqual({
    topK: true,
    forcedReplay: true,
    replayScoring: true,
    tokenizer: true,
    namedRoles: false,
    userSeatGeneration: true,
    sceneStitching: true,
  });

  result["conversation"].supportsDrowseUserSeatGeneration.mockReturnValue(
    false,
  );
  expect(result.getDrowseRuntimeCapabilities().sceneStitching).toBe(false);
});

test("residual capture assembles layer-major positions across prefill chunks", async () => {
  const result = pipeline();
  result["drowseCapturePrefill"] = jest.fn();
  result["drowseCaptureDecoding"] = jest.fn();
  result["prefillChunkSize"] = 2;
  result["contextWindowSize"] = 8;
  result["slidingWindowSize"] = -1;
  result["resetDrowseCaptureState"] = jest.fn();
  result["embedAndForward"] = jest.fn(async () => tensor());
  result["captureDrowseTokenChunk"] = jest
    .fn<(...args: any[]) => Promise<Float32Array>>()
    .mockResolvedValueOnce(Float32Array.from([10, 11, 12, 13, 20, 21, 22, 23]))
    .mockResolvedValueOnce(Float32Array.from([30, 31, 32, 33, 40, 41, 42, 43]));

  const capture = await result.captureDrowseResiduals(
    [100, 101, 102, 103, 104],
    [0, 4],
  );

  expect(result["embedAndForward"]).toHaveBeenCalledWith(
    [[102, 103]],
    2,
    false,
  );
  expect(result["captureDrowseTokenChunk"]).toHaveBeenNthCalledWith(
    1,
    [100, 101],
    [0],
  );
  expect(result["captureDrowseTokenChunk"]).toHaveBeenNthCalledWith(
    2,
    [104],
    [0],
  );
  expect(result["resetDrowseCaptureState"]).toHaveBeenCalledTimes(2);
  expect(capture).toEqual({
    layerCount: 2,
    positionCount: 2,
    hiddenSize: 4,
    positions: [0, 4],
    values: Float32Array.from([
      10, 11, 12, 13, 30, 31, 32, 33, 20, 21, 22, 23, 40, 41, 42, 43,
    ]),
  });
});

test("residual capture resets KV state after a failed chunk", async () => {
  const result = pipeline();
  result["drowseCapturePrefill"] = jest.fn();
  result["drowseCaptureDecoding"] = jest.fn();
  result["prefillChunkSize"] = 2;
  result["contextWindowSize"] = 8;
  result["slidingWindowSize"] = -1;
  result["resetDrowseCaptureState"] = jest.fn();
  result["captureDrowseTokenChunk"] = jest.fn(async () => {
    throw new Error("device lost");
  });

  await expect(result.captureDrowseResiduals([100, 101], [1])).rejects.toThrow(
    "device lost",
  );
  expect(result["resetDrowseCaptureState"]).toHaveBeenCalledTimes(2);
});

test("capture row preparation renders the model chat template and pools before its separator", () => {
  const result = pipeline();
  result["config"] = {
    model_config: { hidden_size: 4, num_hidden_layers: 2 },
    conv_template: {
      system_template: "S:{system_message}<e>",
      system_message: "default",
      roles: {
        user: "<|im_start|>user",
        assistant: "<|im_start|>assistant",
      },
      seps: ["<e>"],
      stop_str: [],
      stop_token_ids: [999],
      system_prefix_token_ids: [7],
    },
  };
  result["tokenizer"] = {
    encode(text: string) {
      const suffix = text.endsWith("<e>");
      const content = suffix ? text.slice(0, -3) : text;
      return Int32Array.from([
        ...Array.from(content, (character) => character.codePointAt(0)!),
        ...(suffix ? [999] : []),
      ]);
    },
  };

  const [prepared] = result.prepareDrowseCaptureRows(
    [
      {
        system: "brief",
        messages: [
          { role: "user", content: "prompt" },
          { role: "system", content: "keep this position" },
          { role: "assistant", content: "answer", roleName: "Guide" },
        ],
      },
    ],
    [999],
  );

  expect(prepared.inputIds[0]).toBe(7);
  expect(prepared.inputIds.at(-1)).toBe(999);
  expect(prepared.inputIds[prepared.position]).toBe("r".codePointAt(0));
  expect(prepared.position).toBe(prepared.inputIds.length - 2);
  expect(
    prepared.inputIds
      .slice(1)
      .map((tokenId: number) =>
        tokenId === 999 ? "<e>" : String.fromCodePoint(tokenId),
      )
      .join(""),
  ).toBe(
    "S:brief<e>" +
      "<|im_start|>user: prompt<e>" +
      "S:keep this position<e>" +
      "<|im_start|>Guide: answer<e>",
  );
});

test("failed Drowse GPU allocation closes its TVM scope and preserves the old program", () => {
  const result = pipeline();
  const previous = { enabled: tensor() };
  result["drowseProgram"] = previous;
  result["tvm"].empty
    .mockImplementationOnce(() => tensor())
    .mockImplementationOnce(() => tensor())
    .mockImplementationOnce(() => {
      throw new Error("allocation failed");
    });

  expect(() => result.setDrowseRankOneProgram(program())).toThrow(
    "allocation failed",
  );
  expect(result["tvm"].beginScope).toHaveBeenCalledTimes(1);
  expect(result["tvm"].endScope).toHaveBeenCalledTimes(1);
  expect(result["drowseProgram"]).toBe(previous);
  expect(previous.enabled.dispose).not.toHaveBeenCalled();
});

test("Drowse program promotion detaches every buffer before disposing the old program", () => {
  const result = pipeline();
  const previous = { enabled: tensor() };
  result["drowseProgram"] = previous;

  result.setDrowseRankOneProgram(program());

  expect(result["tvm"].detachFromCurrentScope).toHaveBeenCalledTimes(8);
  expect(result["tvm"].endScope).toHaveBeenCalledTimes(1);
  expect(previous.enabled.dispose).toHaveBeenCalledTimes(1);
  expect(Object.keys(result["drowseProgram"])).toHaveLength(8);
});

test("measurement read closes its TVM scope when device synchronization fails", async () => {
  const result = pipeline();
  result["drowseMeasurements"] = tensor();
  result["device"].sync.mockRejectedValueOnce(new Error("device lost"));

  await expect(result.readDrowseMeasurements()).rejects.toThrow("device lost");
  expect(result["tvm"].beginScope).toHaveBeenCalledTimes(1);
  expect(result["tvm"].endScope).toHaveBeenCalledTimes(1);
});

test("geometry measurement read returns a detached host copy", async () => {
  const result = pipeline();
  result["drowseGeometryMeasurements"] = {
    ...tensor(),
    shape: [1, 1, 4],
  };
  const host = {
    ...tensor(),
    shape: [1, 1, 4],
    toArray: jest.fn(() => Float32Array.from([1, 0.5, 0.25, 0.75])),
  };
  result["tvm"].empty.mockReturnValueOnce(host);

  await expect(result.readDrowseGeometryMeasurements()).resolves.toEqual(
    Float32Array.from([1, 0.5, 0.25, 0.75]),
  );
  expect(result["tvm"].beginScope).toHaveBeenCalledTimes(1);
  expect(result["tvm"].endScope).toHaveBeenCalledTimes(1);
});

test("one synchronization materializes every token readout before queued control changes", async () => {
  const result = pipeline();
  const dataTensor = (shape: number[], data: Float32Array | Int32Array) => ({
    ...tensor(),
    shape,
    data,
    toArray: jest.fn(() => data),
  });
  result["drowseMeasurements"] = dataTensor(
    [2, 8],
    Float32Array.from({ length: 16 }, (_value, index) => index),
  );
  result["drowseGeometryMeasurements"] = dataTensor(
    [2, 1, 4],
    Float32Array.from({ length: 8 }, (_value, index) => index / 10),
  );
  result["drowseProbeKindHost"] = Uint32Array.from([
    3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  ]);
  const selectedHost = dataTensor(
    [1, 8],
    Float32Array.from([0.25, 0, 0, 0, 0, 0, 0, 0]),
  );
  const layerTokenIdsHost = dataTensor(
    [1, 8],
    Int32Array.from([7, 6, 5, 4, 3, 2, 1, 0]),
  );
  const layerProbabilitiesHost = dataTensor(
    [1, 8],
    Float32Array.from([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
  );
  result["drowsePendingJlensReadback"] = {
    layerCount: 2,
    chunks: [
      {
        chunk: {
          layerIds: Int32Array.of(0),
          jacobians: tensor(),
          layerIdsDevice: tensor(),
        },
        selectedHost,
        layerTokenIdsHost,
        layerProbabilitiesHost,
      },
    ],
    aggregate: {
      tokenIdsHost: dataTensor(
        [1, 8],
        Int32Array.from([7, 6, 5, 4, 3, 2, 1, 0]),
      ),
      statsHost: dataTensor(
        [3, 8],
        Float32Array.from({ length: 24 }, (_value, index) => index / 100),
      ),
      fittedLayerCount: 1,
    },
  };
  result["drowsePendingSaeReadback"] = {
    valuesHost: dataTensor([1, 8], Float32Array.from([9, 8, 7, 6, 5, 4, 3, 2])),
    featureIdsHost: dataTensor(
      [1, 8],
      Int32Array.from([8, 3, 10, 1, 11, 0, 9, 2]),
    ),
    runtimeLayerIndex: 1,
    featureCount: 10,
  };
  result["tvm"].empty.mockImplementation(
    (shape: number[], _dtype: string, device: unknown) => {
      const host: any = dataTensor(shape, new Float32Array(0));
      host.copyFrom = jest.fn((source: any) => {
        host.data = source.data;
        host.toArray = jest.fn(() => host.data);
        return host;
      });
      return device === "cpu" ? host : { ...tensor(), shape };
    },
  );
  const affineActive = tensor();
  const affineAlong = tensor();
  result["drowseStructuredProgram"] = { affineActive, affineAlong };
  result["drowseAffineAlongHost"] = new Float32Array(8);

  const bundle = await result.readDrowseMeasurementBundle();
  expect(bundle.scalar?.[0]).toBeCloseTo(0.25);
  expect(bundle.geometry).toEqual(
    Float32Array.from({ length: 8 }, (_value, index) => index / 10),
  );
  expect(bundle.jlensTopTokens?.tokenIds).toEqual(
    Int32Array.from([7, 6, 5, 4, 3, 2, 1, 0]),
  );
  expect(bundle.saeTopFeatures?.featureIds).toEqual(
    Int32Array.from([8, 3, 1, 0, 9, 2]),
  );
  expect(result["device"].sync).toHaveBeenCalledTimes(1);

  await result.updateDrowseStructuredControls(
    Uint32Array.from([1, 0, 0, 0, 0, 0, 0, 0]),
    new Uint32Array(8),
  );
  expect(affineActive.copyFrom).toHaveBeenCalled();
  expect(affineAlong.copyFrom).toHaveBeenCalled();
  expect(result["device"].sync).toHaveBeenCalledTimes(1);
});

test("J-lens resolver returns exact selected LM-head directions and validates IDs", async () => {
  const result = pipeline();
  result["fullVocabSize"] = 64;
  result["params"] = { model: true };
  result["drowseExactReadoutAttested"] = true;
  result["drowseJlensReadoutAccumulate"] = jest.fn();
  result["drowseJlensReadoutTopK"] = jest.fn();
  result["drowseJlensDirections"] = jest.fn(
    (jacobian: { shape: number[] }) => ({
      ...tensor(),
      shape: [jacobian.shape[0], 8, 4],
    }),
  );
  let directionReadOffset = 0;
  result["tvm"].empty.mockImplementation(
    (shape: number[], _dtype: string, device: unknown) => {
      if (device !== "cpu") return { ...tensor(), shape };
      const length = shape.reduce((product, size) => product * size, 1);
      const offset = directionReadOffset;
      directionReadOffset += length;
      return {
        ...tensor(),
        shape,
        toArray: jest.fn(() =>
          Float32Array.from({ length }, (_value, index) => offset + index),
        ),
      };
    },
  );

  await result.setDrowseJlensDictionary({
    hookAbi: DROWSE_HOOK_ABI,
    bindingId: JLENS_BINDING,
    hiddenSize: 4,
    layerIndices: Int32Array.from([0, 1]),
    matrices: [new Float32Array(16), new Float32Array(16)],
  });
  expect(result["device"].sync).toHaveBeenCalledTimes(1);
  result["device"].sync.mockClear();
  result["tvm"].endScope.mockClear();

  await expect(
    result.resolveDrowseJlensTokenDirections(JLENS_BINDING, [0, 1], [3, 7]),
  ).resolves.toEqual(
    Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39]),
  );
  expect(result["drowseJlensDirections"]).toHaveBeenCalledTimes(2);
  expect(result["drowseJlensDirections"].mock.calls[0][0].shape).toEqual([
    1, 4, 4,
  ]);
  expect(result["drowseJlensDirections"].mock.calls[1][0].shape).toEqual([
    1, 4, 4,
  ]);
  expect(result["device"].sync).toHaveBeenCalledTimes(1);
  expect(result["tvm"].endScope).toHaveBeenCalledTimes(1);

  directionReadOffset = 0;
  result["drowseJlensDirections"].mockClear();
  result["tvm"].endScope.mockClear();
  result["device"].sync.mockClear();

  await expect(
    result.resolveDrowseJlensTokenDirections(JLENS_BINDING, [1], [3, 7]),
  ).resolves.toEqual(Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7]));
  expect(result["drowseJlensDirections"]).toHaveBeenCalledTimes(1);
  expect(result["device"].sync).toHaveBeenCalledTimes(1);
  expect(result["tvm"].endScope).toHaveBeenCalledTimes(1);

  await expect(
    result.resolveDrowseJlensTokenDirections(JLENS_BINDING, [0, 1], []),
  ).rejects.toThrow("token IDs are invalid");
  await expect(
    result.resolveDrowseJlensTokenDirections(JLENS_BINDING, [0, 1], [64]),
  ).rejects.toThrow("token IDs are invalid");
  await expect(
    result.resolveDrowseJlensTokenDirections("c".repeat(64), [0], [3]),
  ).rejects.toThrow("does not match the resident dictionary");
  await expect(
    result.resolveDrowseJlensTokenDirections(JLENS_BINDING, [2], [3]),
  ).rejects.toThrow("no matrix for runtime layer 2");
});

test("resident J-lens replacement is atomic and rejects binding collisions", async () => {
  const result = pipeline();
  result["drowseExactReadoutAttested"] = true;
  result["drowseJlensReadoutAccumulate"] = jest.fn();
  result["drowseJlensReadoutTopK"] = jest.fn();
  const dictionary = {
    hookAbi: DROWSE_HOOK_ABI,
    bindingId: JLENS_BINDING,
    hiddenSize: 4,
    layerIndices: Int32Array.from([0, 1]),
    matrices: [new Float32Array(16), new Float32Array(16)],
  } as const;
  await result.setDrowseJlensDictionary(dictionary);
  const resident = result["drowseJlensDictionary"];

  await expect(
    result.setDrowseJlensDictionary({
      ...dictionary,
      layerIndices: Int32Array.from([0]),
      matrices: [new Float32Array(16)],
    }),
  ).rejects.toThrow(/conflicting metadata/);
  expect(result["drowseJlensDictionary"]).toBe(resident);

  result["device"].sync.mockRejectedValueOnce(new Error("device lost"));
  await expect(
    result.setDrowseJlensDictionary({
      ...dictionary,
      bindingId: "c".repeat(64),
    }),
  ).rejects.toThrow("device lost");
  expect(result["drowseJlensDictionary"]).toBe(resident);
  for (const chunk of resident.chunks) {
    expect(chunk.jacobians.dispose).not.toHaveBeenCalled();
    expect(chunk.layerIdsDevice.dispose).not.toHaveBeenCalled();
  }
});

test("J-lens chunk planning caps Smol and Qwen allocations at 32 MiB", () => {
  const limits = {
    maxBufferSize: 1 << 28,
    maxStorageBufferBindingSize: 1 << 27,
  };
  const smol = planDrowseJlensChunks(
    Array.from({ length: 31 }, (_value, layer) => layer),
    960,
    limits,
  );
  expect(smol.map(({ layerIds }) => layerIds.length)).toEqual(
    Array.from({ length: 31 }, () => 1),
  );
  const qwen17 = planDrowseJlensChunks(
    Array.from({ length: 28 }, (_value, layer) => layer),
    2048,
    limits,
  );
  expect(qwen17.map(({ layerIds }) => layerIds.length)).toEqual(
    Array.from({ length: 28 }, () => 1),
  );
  const qwen4 = planDrowseJlensChunks(
    Array.from({ length: 36 }, (_value, layer) => layer),
    2560,
    limits,
  );
  expect(qwen4.map(({ layerIds }) => layerIds.length)).toEqual(
    Array.from({ length: 36 }, () => 1),
  );
  for (const chunk of [...smol, ...qwen17, ...qwen4]) {
    expect(chunk.byteLength).toBeLessThanOrEqual(DROWSE_JLENS_MAX_CHUNK_BYTES);
    expect(chunk.byteLength).toBeLessThanOrEqual(limits.maxBufferSize);
    expect(chunk.byteLength).toBeLessThanOrEqual(
      limits.maxStorageBufferBindingSize,
    );
  }
  expect(() =>
    planDrowseJlensChunks([0], 2048, {
      maxBufferSize: 2048 * 2048 * 4 - 1,
      maxStorageBufferBindingSize: 1 << 27,
    }),
  ).toThrow(/exceeding the WebGPU buffer limit/);
  expect(() => planDrowseJlensChunks([0], 3000, limits)).toThrow(
    /exceeding the WebGPU buffer limit 33554432/,
  );
  expect(() => planDrowseJlensChunks([1, 1], 4, limits)).toThrow(
    /layer IDs are invalid/,
  );
});

test("J-lens probability readout reuses resident multi-layer chunks and disposes them", async () => {
  const result = pipeline();
  result["config"].model_config.num_hidden_layers = 4;
  result["params"] = { model: true };
  result["drowseProbeKindHost"] = Uint32Array.from([
    3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3,
    0, 0, 0, 0, 0, 0, 0,
  ]);
  const plans = planDrowseJlensChunks([0, 2, 3], 4, {
    maxBufferSize: 128,
    maxStorageBufferBindingSize: 128,
  });
  const chunks = await result["uploadDrowseJlensChunks"](
    Array.from({ length: 3 }, (_value, matrix) =>
      Float32Array.from(
        { length: 4 * 4 },
        (_entry, index) => matrix * 16 + index,
      ),
    ),
    Int32Array.from([0, 2, 3]),
    plans,
    4,
  );
  expect(chunks).toHaveLength(3);
  expect(chunks[0].jacobians.shape).toEqual([1, 4, 4]);
  expect(chunks[1].jacobians.shape).toEqual([1, 4, 4]);
  expect(chunks[2].jacobians.shape).toEqual([1, 4, 4]);
  expect(chunks[0].jacobians.copyFrom).toHaveBeenCalledTimes(1);
  expect(chunks[1].jacobians.copyFrom).toHaveBeenCalledTimes(1);
  expect(chunks[2].jacobians.copyFrom).toHaveBeenCalledTimes(1);
  expect(chunks[0].layerIdsDevice.copyFrom).toHaveBeenCalledWith(
    Int32Array.from([0]),
  );
  expect(chunks[1].layerIdsDevice.copyFrom).toHaveBeenCalledWith(
    Int32Array.from([2]),
  );
  expect(chunks[2].layerIdsDevice.copyFrom).toHaveBeenCalledWith(
    Int32Array.from([3]),
  );
  expect(chunks[0].jacobians.copyFrom).toHaveBeenCalledWith(
    Float32Array.from({ length: 4 * 4 }, (_value, index) => index),
  );
  expect(chunks[1].jacobians.copyFrom).toHaveBeenCalledWith(
    Float32Array.from({ length: 4 * 4 }, (_value, index) => 4 * 4 + index),
  );
  expect(chunks[2].jacobians.copyFrom).toHaveBeenCalledWith(
    Float32Array.from({ length: 4 * 4 }, (_value, index) => 2 * 4 * 4 + index),
  );
  expect(result["device"].sync).toHaveBeenCalledTimes(1);
  result["device"].sync.mockClear();
  result["drowseJlensDictionary"] = { bindingId: JLENS_BINDING, chunks };
  result["drowseJlensChunks"] = chunks;
  result["drowseJlensReadout"] = jest.fn(
    (_hidden, jacobians: { shape: number[] }) => ({
      ...tensor(),
      shape: [jacobians.shape[0], 8],
    }),
  );
  result["tvm"].empty.mockClear();
  let readOffset = 0;
  result["tvm"].empty.mockImplementation(
    (shape: number[], _dtype: string, device: unknown) => {
      if (device !== "cpu") return { ...tensor(), shape };
      const offset = readOffset;
      readOffset += shape[0] * 8;
      return {
        ...tensor(),
        shape,
        toArray: jest.fn(() =>
          Float32Array.from(
            { length: shape[0] * 8 },
            (_value, index) => offset + index,
          ),
        ),
      };
    },
  );

  await result["computeDrowseJlensProbabilities"](tensor(), tensor());
  expect(result["device"].sync).not.toHaveBeenCalled();
  await result.readDrowseMeasurementBundle();
  expect(result["drowseJlensProbabilitiesHost"]).toEqual(
    Float32Array.from([
      0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14,
      15, 16, 17, 18, 19, 20, 21, 22, 23,
    ]),
  );
  expect(result["drowseJlensReadout"]).toHaveBeenCalledTimes(3);
  expect(
    result["tvm"].empty.mock.calls.every(
      ([, , device]: [number[], string, unknown]) => device === "cpu",
    ),
  ).toBe(true);
  expect(
    result["tvm"].empty.mock.calls.filter(
      ([shape]: [number[]]) =>
        JSON.stringify(shape) === JSON.stringify([2, 4, 4]),
    ),
  ).toHaveLength(0);
  for (const chunk of chunks) {
    expect(chunk.jacobians.copyFrom).toHaveBeenCalledTimes(1);
    expect(chunk.layerIdsDevice.copyFrom).toHaveBeenCalledTimes(1);
  }
  expect(result["device"].sync).toHaveBeenCalledTimes(1);
  result.clearDrowseRankOneProgram();
  for (const chunk of chunks) {
    expect(chunk.jacobians.dispose).not.toHaveBeenCalled();
    expect(chunk.layerIdsDevice.dispose).not.toHaveBeenCalled();
  }
  result["drowseExactReadoutAttested"] = true;
  result["drowseJlensReadoutAccumulate"] = jest.fn();
  result["drowseJlensReadoutTopK"] = jest.fn();
  await result.clearDrowseJlensDictionary();
  for (const chunk of chunks) {
    expect(chunk.jacobians.dispose).toHaveBeenCalledTimes(1);
    expect(chunk.layerIdsDevice.dispose).toHaveBeenCalledTimes(1);
  }
});

test("J-lens probability readout rejects an invalid batched result shape", async () => {
  const result = pipeline();
  result["params"] = { model: true };
  result["drowseProbeKindHost"] = new Uint32Array(16);
  result["drowseJlensChunks"] = [
    {
      layerIds: Int32Array.from([0, 1]),
      jacobians: tensor(),
      layerIdsDevice: tensor(),
    },
  ];
  result["drowseJlensReadout"] = jest.fn(() => ({
    ...tensor(),
    shape: [1, 8],
  }));

  await expect(
    result["computeDrowseJlensProbabilities"](tensor(), tensor()),
  ).rejects.toThrow(/probabilities tensor has the wrong shape/);
  expect(result["tvm"].endScope).toHaveBeenCalledTimes(1);
});

test("SAE chunk planning respects WebGPU buffers, compiler bounds, and top-8 padding", () => {
  const limits = {
    maxBufferSize: 1 << 30,
    maxStorageBufferBindingSize: 1 << 30,
  };
  const chunks = planDrowseSaeChunks(4, 20000, limits);
  expect(chunks.map(({ featureCount }) => featureCount)).toEqual([
    DROWSE_SAE_MAX_FEATURES_PER_CHUNK,
    20000 - DROWSE_SAE_MAX_FEATURES_PER_CHUNK,
  ]);
  expect(
    chunks.every(
      ({ encoderByteLength }) =>
        encoderByteLength <= DROWSE_SAE_MAX_CHUNK_BYTES,
    ),
  ).toBe(true);
  expect(planDrowseSaeChunks(4, 3, limits)).toEqual([
    {
      featureOffset: 0,
      featureCount: 3,
      paddedFeatureCount: 8,
      encoderByteLength: 128,
    },
  ]);
  expect(() =>
    planDrowseSaeChunks(4, 8, {
      maxBufferSize: 127,
      maxStorageBufferBindingSize: 127,
    }),
  ).toThrow(/requires at least 128 bytes/);

  const smol = planDrowseSaeChunks(960, 16384, limits);
  expect(smol.map(({ featureCount }) => featureCount)).toEqual([8738, 7646]);
  const qwen06 = planDrowseSaeChunks(1024, 16384, limits);
  expect(qwen06.map(({ featureCount }) => featureCount)).toEqual([8192, 8192]);
  const qwen17 = planDrowseSaeChunks(2048, 32768, limits);
  expect(qwen17.map(({ featureCount }) => featureCount)).toEqual(
    Array.from({ length: 8 }, () => 4096),
  );
  for (const chunk of [...smol, ...qwen06, ...qwen17]) {
    expect(chunk.encoderByteLength).toBeLessThanOrEqual(
      DROWSE_SAE_MAX_CHUNK_BYTES,
    );
  }
});

test("precomputed SAE upload is row-major, bounded, atomic, and disposable", async () => {
  const result = pipeline();
  result["drowseSaeReadoutAccumulate"] = jest.fn();
  result["drowseExactReadoutAttested"] = true;
  result["drowseJlensBufferLimits"] = {
    maxBufferSize: 128,
    maxStorageBufferBindingSize: 128,
  };
  const dictionary = {
    hookAbi: DROWSE_HOOK_ABI,
    bindingId: SAE_BINDING,
    hiddenSize: 4,
    runtimeLayerIndex: 1,
    featureCount: 10,
    activation: "relu",
    encoder: Float32Array.from({ length: 40 }, (_value, index) => index),
    encoderBias: Float32Array.from(
      { length: 10 },
      (_value, index) => 100 + index,
    ),
    encoderThreshold: new Float32Array(10),
    decoderBias: Float32Array.from([1, 2, 3, 4]),
  } as const;

  await result.setDrowseSaeDictionary(dictionary);
  const resident = result["drowseSaeDictionary"];
  expect(resident.runtimeLayerIndex).toBe(1);
  expect(resident.featureCount).toBe(10);
  expect(resident.chunks).toHaveLength(2);
  expect(resident.chunks[0].encoder.shape).toEqual([4, 8]);
  expect(resident.chunks[1].encoder.shape).toEqual([4, 8]);
  expect(resident.chunks[1].encoder.copyFrom).toHaveBeenCalledWith(
    Float32Array.from([
      8, 9, 0, 0, 0, 0, 0, 0, 18, 19, 0, 0, 0, 0, 0, 0, 28, 29, 0, 0, 0, 0, 0,
      0, 38, 39, 0, 0, 0, 0, 0, 0,
    ]),
  );
  expect(resident.chunks[1].encoderBias.copyFrom).toHaveBeenCalledWith(
    Float32Array.from([108, 109, 0, 0, 0, 0, 0, 0]),
  );
  expect(result["device"].sync).toHaveBeenCalledTimes(1);

  await result.clearDrowseSaeDictionary();
  expect(resident.decoderBias.dispose).toHaveBeenCalledTimes(1);
  expect(resident.runtimeLayerIndexDevice.dispose).toHaveBeenCalledTimes(1);
  for (const chunk of resident.chunks) {
    expect(chunk.encoder.dispose).toHaveBeenCalledTimes(1);
    expect(chunk.encoderBias.dispose).toHaveBeenCalledTimes(1);
    expect(chunk.featureOffsetDevice.dispose).toHaveBeenCalledTimes(1);
  }
  await expect(result.readDrowseSaeTopFeatures()).resolves.toBeUndefined();
});

test("resident SAE replacement is atomic and rejects binding collisions", async () => {
  const result = pipeline();
  result["drowseSaeReadoutAccumulate"] = jest.fn();
  result["drowseExactReadoutAttested"] = true;
  const dictionary = {
    hookAbi: DROWSE_HOOK_ABI,
    bindingId: SAE_BINDING,
    hiddenSize: 4,
    runtimeLayerIndex: 1,
    featureCount: 8,
    activation: "relu",
    encoder: new Float32Array(32),
    encoderBias: new Float32Array(8),
    encoderThreshold: new Float32Array(8),
    decoderBias: new Float32Array(4),
  } as const;
  await result.setDrowseSaeDictionary(dictionary);
  const resident = result["drowseSaeDictionary"];

  await expect(
    result.setDrowseSaeDictionary({
      ...dictionary,
      runtimeLayerIndex: 0,
    }),
  ).rejects.toThrow(/conflicting metadata/);
  expect(result["drowseSaeDictionary"]).toBe(resident);

  result["device"].sync.mockRejectedValueOnce(new Error("device lost"));
  await expect(
    result.setDrowseSaeDictionary({
      ...dictionary,
      bindingId: "d".repeat(64),
    }),
  ).rejects.toThrow("device lost");
  expect(result["drowseSaeDictionary"]).toBe(resident);
  expect(resident.decoderBias.dispose).not.toHaveBeenCalled();
  expect(resident.runtimeLayerIndexDevice.dispose).not.toHaveBeenCalled();
});

test("exact SAE readout chains GPU top-8 state and filters padded feature IDs", async () => {
  const result = pipeline();
  result["drowseExactReadoutAttested"] = true;
  const firstValues = { ...tensor(), shape: [1, 8], name: "first-values" };
  const firstIds = { ...tensor(), shape: [1, 8], name: "first-ids" };
  const finalValues = { ...tensor(), shape: [1, 8], name: "final-values" };
  const finalIds = { ...tensor(), shape: [1, 8], name: "final-ids" };
  result["drowseSaeReadoutAccumulate"] = jest
    .fn()
    .mockReturnValueOnce({
      get: (index: number) => [firstValues, firstIds][index],
    })
    .mockReturnValueOnce({
      get: (index: number) => [finalValues, finalIds][index],
    });
  result["drowseSaeDictionary"] = {
    bindingId: SAE_BINDING,
    runtimeLayerIndex: 1,
    featureCount: 10,
    decoderBias: tensor(),
    runtimeLayerIndexDevice: tensor(),
    chunks: [
      {
        featureOffset: 0,
        featureCount: 8,
        paddedFeatureCount: 8,
        encoderByteLength: 128,
        encoder: tensor(),
        encoderBias: tensor(),
        featureOffsetDevice: tensor(),
      },
      {
        featureOffset: 8,
        featureCount: 2,
        paddedFeatureCount: 8,
        encoderByteLength: 128,
        encoder: tensor(),
        encoderBias: tensor(),
        featureOffsetDevice: tensor(),
      },
    ],
  };
  result["drowseSaeReadoutActive"] = true;
  result["tvm"].empty.mockImplementation(
    (shape: number[], dtype: string, device: unknown) => {
      if (device !== "cpu") return { ...tensor(), shape };
      return {
        ...tensor(),
        shape,
        toArray: jest.fn(() =>
          dtype === "int32"
            ? Int32Array.from([8, 3, 10, 1, 11, 0, 9, 2])
            : Float32Array.from([9, 8, 7, 6, 5, 4, 3, 2]),
        ),
      };
    },
  );

  await result["computeDrowseSaeTopFeatures"](tensor());
  expect(result["drowseSaeReadoutAccumulate"]).toHaveBeenCalledTimes(2);
  expect(result["drowseSaeReadoutAccumulate"].mock.calls[1].slice(-2)).toEqual([
    firstValues,
    firstIds,
  ]);
  await expect(result.readDrowseSaeTopFeatures()).resolves.toEqual({
    featureIds: Int32Array.from([8, 3, 1, 0, 9, 2]),
    activations: Float32Array.from([9, 8, 6, 4, 3, 2]),
    runtimeLayerIndex: 1,
    featureCount: 10,
  });
  expect(result["device"].sync).toHaveBeenCalledTimes(1);
});

test("exact J-lens readout preserves selected probes and publishes per-layer and aggregate top-8", async () => {
  const result = pipeline();
  result["config"].model_config.num_hidden_layers = 4;
  result["fullVocabSize"] = 16;
  result["params"] = { model: true };
  result["drowseExactReadoutAttested"] = true;
  result["drowseProbeKindHost"] = new Uint32Array(32);
  result["drowseJlensReadout"] = jest.fn();
  const dataTensor = (
    shape: number[],
    data: Float32Array | Int32Array,
    name: string,
  ) => ({ ...tensor(), shape, data, name });
  const accumulated = [0, 1].map((index) => {
    const selected = dataTensor(
      [1, 8],
      Float32Array.from({ length: 8 }, (_value, column) => index * 8 + column),
      `selected-${index}`,
    );
    const layerTokenIds = dataTensor(
      [1, 8],
      Int32Array.from({ length: 8 }, (_value, column) => index * 8 + column),
      `layer-ids-${index}`,
    );
    const layerProbabilities = dataTensor(
      [1, 8],
      Float32Array.from({ length: 8 }, (_value, column) => index + column / 10),
      `layer-probs-${index}`,
    );
    const probabilitySum = dataTensor(
      [16],
      new Float32Array(16),
      `sum-${index}`,
    );
    const depthSum = dataTensor([16], new Float32Array(16), `depth-${index}`);
    const depthSquareSum = dataTensor(
      [16],
      new Float32Array(16),
      `depth-square-${index}`,
    );
    return {
      get: (position: number) =>
        [
          selected,
          layerTokenIds,
          layerProbabilities,
          probabilitySum,
          depthSum,
          depthSquareSum,
        ][position],
      probabilitySum,
      depthSum,
      depthSquareSum,
    };
  });
  result["drowseJlensReadoutAccumulate"] = jest
    .fn()
    .mockReturnValueOnce(accumulated[0])
    .mockReturnValueOnce(accumulated[1]);
  result["drowseJlensReadoutTopK"] = jest.fn(() => {
    expect(result["device"].sync).not.toHaveBeenCalled();
    return {
      get: (position: number) =>
        [
          dataTensor(
            [1, 8],
            Int32Array.from([7, 6, 5, 4, 3, 2, 1, 0]),
            "aggregate-ids",
          ),
          dataTensor(
            [3, 8],
            Float32Array.from([
              0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, 0.1, 0.2, 0.3, 0.4,
              0.5, 0.6, 0.7, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
            ]),
            "aggregate-stats",
          ),
        ][position],
    };
  });
  result["drowseJlensChunks"] = [0, 2].map((layerId) => ({
    layerIds: Int32Array.of(layerId),
    jacobians: tensor(),
    layerIdsDevice: tensor(),
  }));
  result["tvm"].empty.mockImplementation((shape: number[], dtype: string) => {
    const host: any = {
      ...tensor(),
      shape,
      data:
        dtype === "int32"
          ? new Int32Array(shape.reduce((a, b) => a * b, 1))
          : new Float32Array(shape.reduce((a, b) => a * b, 1)),
    };
    host.copyFrom = jest.fn((source: any) => {
      host.data = source?.data ?? source;
      return host;
    });
    host.toArray = jest.fn(() => host.data);
    return host;
  });

  await result["computeDrowseJlensProbabilities"](tensor(), tensor());
  expect(result["device"].sync).not.toHaveBeenCalled();
  const bundle = await result.readDrowseMeasurementBundle();
  expect(
    Array.from(result["drowseJlensProbabilitiesHost"].slice(0, 8)),
  ).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  expect(
    Array.from(result["drowseJlensProbabilitiesHost"].slice(16, 24)),
  ).toEqual([8, 9, 10, 11, 12, 13, 14, 15]);
  expect(result["drowseJlensReadout"]).not.toHaveBeenCalled();
  expect(result["drowseJlensReadoutAccumulate"]).toHaveBeenCalledTimes(2);
  expect(
    result["drowseJlensReadoutAccumulate"].mock.calls[1].slice(4, 7),
  ).toEqual([
    accumulated[0].probabilitySum,
    accumulated[0].depthSum,
    accumulated[0].depthSquareSum,
  ]);
  expect(bundle.jlensTopTokens).toEqual({
    tokenIds: Int32Array.from([7, 6, 5, 4, 3, 2, 1, 0]),
    strength: Float32Array.from([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
    centerOfMass: Float32Array.from([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    spread: Float32Array.from([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
    fittedLayerCount: 2,
    layerIndices: Int32Array.from([0, 2]),
    layerTokenIds: Int32Array.from([
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    ]),
    layerProbabilities: Float32Array.from([
      0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
      1.7,
    ]),
  });
  await expect(result.readDrowseJlensTopTokens()).resolves.toEqual(
    bundle.jlensTopTokens,
  );
  expect(result["device"].sync).toHaveBeenCalledTimes(1);
});

test("exact J-lens readout identifies the failing GPU chunk", async () => {
  const result = pipeline();
  result["config"].model_config.num_hidden_layers = 4;
  result["fullVocabSize"] = 16;
  result["params"] = { model: true };
  result["drowseExactReadoutAttested"] = true;
  result["drowseProbeKindHost"] = new Uint32Array(32);
  result["drowseJlensReadout"] = jest.fn();
  result["drowseJlensReadoutAccumulate"] = jest.fn(() => {
    throw { message: "ExitStatus exit(1)" };
  });
  result["drowseJlensReadoutTopK"] = jest.fn();
  result["drowseJlensChunks"] = [
    {
      layerIds: Int32Array.of(2),
      jacobians: tensor(),
      layerIdsDevice: tensor(),
    },
  ];

  await expect(
    result["computeDrowseJlensProbabilities"](tensor(), tensor()),
  ).rejects.toThrow(
    "Drowse exact J-lens failed while running chunk 1/1: ExitStatus exit(1)",
  );
  expect(result["tvm"].endScope).toHaveBeenCalledTimes(1);
});

test("new readout APIs reject VM symbols without the v3 exact-readout attestation", async () => {
  const result = pipeline();
  result["drowseJlensReadoutAccumulate"] = jest.fn();
  result["drowseJlensReadoutTopK"] = jest.fn();
  result["drowseSaeReadoutAccumulate"] = jest.fn();
  result.getDrowseStructuredHookProfile = jest.fn(async () => ({
    exactReadoutAbi: "legacy-readout",
    readoutTopK: 8,
    maxSaeFeaturesPerChunk: 16384,
  }));

  await expect(result.readDrowseJlensTopTokens()).rejects.toThrow(
    /does not attest the exact Drowse readout ABI/,
  );
  await expect(result.readDrowseSaeTopFeatures()).rejects.toThrow(
    /does not attest the exact Drowse readout ABI/,
  );
  expect(result["drowseExactReadoutAttested"]).not.toBe(true);
});
