import { UnknownMessageKindError } from "../src/error";
import {
  CreateWebWorkerMLCEngine,
  WebWorkerMLCEngine,
  WebWorkerMLCEngineHandler,
} from "../src/web_worker";
import { jest, test, expect, beforeEach } from "@jest/globals";

const reloadMock = jest.fn<(...args: any[]) => Promise<void>>(
  async () => undefined,
);
const forwardMock = jest.fn<(...args: any[]) => Promise<any>>();
const chatCompletionMock = jest.fn<(...args: any[]) => Promise<any>>();
const completionMock = jest.fn<(...args: any[]) => Promise<any>>();
const embeddingMock = jest.fn<(...args: any[]) => Promise<any>>();
const setLogitRegistryMock = jest.fn<(...args: any[]) => void>();
const setAppConfigMock = jest.fn<(...args: any[]) => void>();
const supportsDrowseMock = jest.fn<(...args: any[]) => Promise<boolean>>();
const setDrowseMock = jest.fn<(...args: any[]) => Promise<void>>();
const clearDrowseMock = jest.fn<(...args: any[]) => Promise<void>>();
const readDrowseMock = jest.fn<(...args: any[]) => Promise<Float32Array>>();
const supportsDrowseCaptureMock =
  jest.fn<(...args: any[]) => Promise<boolean>>();
const captureDrowseMock = jest.fn<(...args: any[]) => Promise<any>>();
const prepareDrowseCaptureMock = jest.fn<(...args: any[]) => Promise<any>>();
const drowseCapabilitiesMock = jest.fn<(...args: any[]) => Promise<any>>();
const drowseProfileMock = jest.fn<(...args: any[]) => Promise<any>>();
const tokenizeDrowseMock = jest.fn<(...args: any[]) => Promise<number[]>>();
const decodeDrowseMock = jest.fn<(...args: any[]) => Promise<string>>();
const setDrowseSaeMock = jest.fn<(...args: any[]) => Promise<void>>();
const clearDrowseSaeMock = jest.fn<(...args: any[]) => Promise<void>>();
const setDrowseJlensMock = jest.fn<(...args: any[]) => Promise<void>>();
const clearDrowseJlensMock = jest.fn<(...args: any[]) => Promise<void>>();
const resolveDrowseJlensMock =
  jest.fn<(...args: any[]) => Promise<Float32Array>>();
const readDrowseJlensTopMock = jest.fn<(...args: any[]) => Promise<any>>();
const readDrowseSaeTopMock = jest.fn<(...args: any[]) => Promise<any>>();

const mockEngineInstance: Record<string, any> = {
  reload: reloadMock,
  forwardTokensAndSample: forwardMock,
  chatCompletion: chatCompletionMock,
  completion: completionMock,
  embedding: embeddingMock,
  setInitProgressCallback: jest.fn((cb) => {
    mockEngineInstance.__initCb = cb;
  }),
  setLogitProcessorRegistry: setLogitRegistryMock,
  setAppConfig: setAppConfigMock,
  supportsDrowseRankOneHooks: supportsDrowseMock,
  setDrowseRankOneProgram: setDrowseMock,
  setDrowseStructuredProgram: setDrowseMock,
  clearDrowseRankOneProgram: clearDrowseMock,
  readDrowseMeasurements: readDrowseMock,
  supportsDrowseResidualCapture: supportsDrowseCaptureMock,
  captureDrowseResiduals: captureDrowseMock,
  prepareDrowseCaptureRows: prepareDrowseCaptureMock,
  getDrowseRuntimeCapabilities: drowseCapabilitiesMock,
  getDrowseStructuredHookProfile: drowseProfileMock,
  tokenizeDrowseText: tokenizeDrowseMock,
  decodeDrowseTokens: decodeDrowseMock,
  setDrowseSaeDictionary: setDrowseSaeMock,
  clearDrowseSaeDictionary: clearDrowseSaeMock,
  setDrowseJlensDictionary: setDrowseJlensMock,
  clearDrowseJlensDictionary: clearDrowseJlensMock,
  resolveDrowseJlensTokenDirections: resolveDrowseJlensMock,
  readDrowseJlensTopTokens: readDrowseJlensTopMock,
  readDrowseSaeTopFeatures: readDrowseSaeTopMock,
};

jest.mock("../src/engine", () => {
  return {
    MLCEngine: jest.fn(() => mockEngineInstance),
  };
});

beforeEach(() => {
  reloadMock.mockClear();
  forwardMock.mockClear();
  chatCompletionMock.mockClear();
  completionMock.mockClear();
  embeddingMock.mockClear();
  setLogitRegistryMock.mockClear();
  setAppConfigMock.mockClear();
  supportsDrowseMock.mockClear();
  setDrowseMock.mockClear();
  clearDrowseMock.mockClear();
  readDrowseMock.mockClear();
  supportsDrowseCaptureMock.mockClear();
  captureDrowseMock.mockClear();
  prepareDrowseCaptureMock.mockClear();
  drowseCapabilitiesMock.mockClear();
  drowseProfileMock.mockClear();
  tokenizeDrowseMock.mockClear();
  decodeDrowseMock.mockClear();
  setDrowseSaeMock.mockClear();
  clearDrowseSaeMock.mockClear();
  setDrowseJlensMock.mockClear();
  clearDrowseJlensMock.mockClear();
  resolveDrowseJlensMock.mockClear();
  readDrowseJlensTopMock.mockClear();
  readDrowseSaeTopMock.mockClear();
  mockEngineInstance.__initCb = undefined;
  (globalThis as any).postMessage = jest.fn();
});

function flushMicrotasks() {
  return new Promise<void>((resolve) => setTimeout(resolve, 0));
}

test("constructor registers init progress callback and posts updates", () => {
  const handler = new WebWorkerMLCEngineHandler();
  expect(mockEngineInstance.setInitProgressCallback).toHaveBeenCalled();
  const report = { progress: 0.5 };
  mockEngineInstance.__initCb(report);
  expect(globalThis.postMessage).toHaveBeenCalledWith({
    kind: "initProgressCallback",
    uuid: "",
    content: report,
  });
  // suppress unused
  expect(handler).toBeTruthy();
});

test("constructor accepts a caller-configured engine", () => {
  const suppliedEngine = {
    ...mockEngineInstance,
    setInitProgressCallback: jest.fn(),
  };
  const handler = new WebWorkerMLCEngineHandler(suppliedEngine as any);
  expect(handler.engine).toBe(suppliedEngine);
  expect(suppliedEngine.setInitProgressCallback).toHaveBeenCalled();
});

test("worker errors retain nested WebGPU diagnostics", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  setDrowseMock.mockRejectedValueOnce({
    error: {
      name: "GPUOutOfMemoryError",
      message: "allocation exceeded the device limit",
    },
    reason: "out-of-memory",
  });
  handler.onmessage({
    kind: "setDrowseStructuredProgram",
    uuid: "structured-error",
    content: { program: {}, modelId: "demo" },
  });
  await flushMicrotasks();
  expect(globalThis.postMessage).toHaveBeenCalledWith({
    kind: "throw",
    uuid: "structured-error",
    content: expect.stringContaining(
      "GPUOutOfMemoryError: allocation exceeded the device limit",
    ),
  });
});

test("residual capture messages return typed capture data", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  const capture = {
    layerCount: 2,
    positionCount: 1,
    hiddenSize: 2,
    positions: [3],
    values: Float32Array.from([1, 2, 3, 4]),
  };
  captureDrowseMock.mockResolvedValueOnce(capture);
  const onComplete = jest.fn();
  handler.onmessage(
    {
      kind: "captureDrowseResiduals",
      uuid: "capture-1",
      content: { inputIds: [4, 5, 6, 7], positions: [3], modelId: "demo" },
    },
    onComplete,
  );
  await flushMicrotasks();
  expect(captureDrowseMock).toHaveBeenCalledWith([4, 5, 6, 7], [3], "demo");
  expect(onComplete).toHaveBeenCalledWith(capture);
});

test("capture row preparation messages route through the worker", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  const prepared = [{ inputIds: [1, 2, 3], position: 1 }];
  prepareDrowseCaptureMock.mockResolvedValueOnce(prepared);
  const rows = [
    {
      system: "brief",
      messages: [{ role: "assistant", content: "response" }],
    },
  ];
  const onComplete = jest.fn();
  handler.onmessage(
    {
      kind: "prepareDrowseCaptureRows",
      uuid: "prepare-1",
      content: { rows, specialTokenIds: [3], modelId: "demo" },
    },
    onComplete,
  );
  await flushMicrotasks();
  expect(prepareDrowseCaptureMock).toHaveBeenCalledWith(rows, [3], "demo");
  expect(onComplete).toHaveBeenCalledWith(prepared);
});

test("Drowse capability and tokenizer messages route through the worker", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  const capabilities = {
    topK: true,
    forcedReplay: true,
    replayScoring: true,
    tokenizer: true,
    namedRoles: false,
    userSeatGeneration: true,
    sceneStitching: true,
  };
  drowseCapabilitiesMock.mockResolvedValueOnce(capabilities);
  tokenizeDrowseMock.mockResolvedValueOnce([31, 32]);
  decodeDrowseMock.mockResolvedValueOnce(" leading");
  const complete = jest.fn();

  handler.onmessage(
    {
      kind: "getDrowseRuntimeCapabilities",
      uuid: "capabilities",
      content: { modelId: "demo" },
    } as any,
    complete,
  );
  handler.onmessage(
    {
      kind: "tokenizeDrowseText",
      uuid: "tokenize",
      content: { text: " leading", modelId: "demo" },
    } as any,
    complete,
  );
  handler.onmessage(
    {
      kind: "decodeDrowseTokens",
      uuid: "decode",
      content: { tokenIds: [31, 32], modelId: "demo" },
    } as any,
    complete,
  );
  await flushMicrotasks();

  expect(drowseCapabilitiesMock).toHaveBeenCalledWith("demo");
  expect(tokenizeDrowseMock).toHaveBeenCalledWith(" leading", "demo");
  expect(decodeDrowseMock).toHaveBeenCalledWith([31, 32], "demo");
  expect(complete).toHaveBeenCalledWith(capabilities);
  expect(complete).toHaveBeenCalledWith([31, 32]);
  expect(complete).toHaveBeenCalledWith(" leading");
});

test("chatCompletionNonStreaming reloads when worker state mismatches", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  chatCompletionMock.mockResolvedValueOnce({ object: "chat.completion" });
  const message = {
    kind: "chatCompletionNonStreaming",
    uuid: "task-1",
    content: {
      modelId: ["demo"],
      chatOpts: [],
      request: { model: "demo", messages: [{ role: "user", content: "hi" }] },
    },
  };
  const onComplete = jest.fn();
  handler.onmessage(message, onComplete);
  await flushMicrotasks();
  expect(reloadMock).toHaveBeenCalledWith(["demo"], []);
  expect(chatCompletionMock).toHaveBeenCalled();
  expect(onComplete).toHaveBeenCalledWith({ object: "chat.completion" });
  expect(globalThis.postMessage).toHaveBeenCalledWith({
    kind: "return",
    uuid: "task-1",
    content: { object: "chat.completion" },
  });
});

test("chatCompletionStreamInit registers async generator", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  async function* generator() {
    yield { object: "chunk" } as any;
  }
  chatCompletionMock.mockResolvedValueOnce(generator());
  const message = {
    kind: "chatCompletionStreamInit",
    uuid: "stream",
    content: {
      modelId: ["demo"],
      selectedModelId: "demo",
      chatOpts: [],
      request: {
        model: "demo",
        messages: [{ role: "user", content: "go" }],
        stream: true,
      },
    },
  };
  handler.onmessage(message, jest.fn());
  await flushMicrotasks();
  expect(
    (handler as any).loadedModelIdToAsyncGenerator.get("demo"),
  ).toBeDefined();
});

test("completionNonStreaming routes to engine completion", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  completionMock.mockResolvedValueOnce({ object: "text_completion" });
  const message = {
    kind: "completionNonStreaming",
    uuid: "comp",
    content: {
      modelId: ["demo"],
      chatOpts: [],
      request: { model: "demo", prompt: "hi" },
    },
  };
  const onComplete = jest.fn();
  handler.onmessage(message, onComplete);
  await flushMicrotasks();
  expect(completionMock).toHaveBeenCalled();
  expect(onComplete).toHaveBeenCalledWith({ object: "text_completion" });
});

test("embedding message reloads if needed and returns embeddings", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  embeddingMock.mockResolvedValueOnce({ object: "list", data: [] });
  const message = {
    kind: "embedding",
    uuid: "embed",
    content: {
      modelId: ["demo"],
      chatOpts: [],
      request: { model: "demo", input: "text" },
    },
  };
  const onComplete = jest.fn();
  handler.onmessage(message, onComplete);
  await flushMicrotasks();
  expect(embeddingMock).toHaveBeenCalledWith({ model: "demo", input: "text" });
  expect(onComplete).toHaveBeenCalledWith({ object: "list", data: [] });
});

test("reloadIfUnmatched triggers reload when model lists differ", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  handler.modelId = ["a"];
  await handler.reloadIfUnmatched(["b"]);
  expect(reloadMock).toHaveBeenCalledWith(["b"], undefined);
  reloadMock.mockClear();
  handler.modelId = ["same"];
  await handler.reloadIfUnmatched(["same"]);
  expect(reloadMock).not.toHaveBeenCalled();
});

test("reloadIfUnmatched records recovered worker state", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  const chatOpts = [{ temperature: 0.5 }];

  await handler.reloadIfUnmatched(["demo"], chatOpts);
  await handler.reloadIfUnmatched(["demo"], chatOpts);

  expect(reloadMock).toHaveBeenCalledTimes(1);
  expect(handler.modelId).toEqual(["demo"]);
  expect(handler.chatOpts).toBe(chatOpts);
});

test("unknown messages invoke onError and throw", () => {
  const handler = new WebWorkerMLCEngineHandler();
  const onError = jest.fn();
  expect(() =>
    handler.onmessage({ kind: "mystery", content: {} }, undefined, onError),
  ).toThrow(UnknownMessageKindError);
  expect(onError).toHaveBeenCalled();
});

test("CreateWebWorkerMLCEngine instantiates client and reloads", async () => {
  const worker = { postMessage: jest.fn(), onmessage: undefined as any };
  const reloadSpy = jest
    .spyOn(WebWorkerMLCEngine.prototype, "reload")
    .mockResolvedValue(undefined);
  const engine = await CreateWebWorkerMLCEngine(worker, "model@a");
  expect(reloadSpy).toHaveBeenCalledWith("model@a", undefined);
  expect(engine.worker).toBe(worker);
  reloadSpy.mockRestore();
});

class MockWorker {
  public sent: any[] = [];
  public onmessage?: (event: any) => void;
  private responders: Map<string, (msg: any) => any> = new Map();

  constructor() {
    this.setResponder("completionNonStreaming", () => ({
      object: "completion",
    }));
    this.setResponder("embedding", () => ({ object: "list", data: [] }));
    this.setResponder("reload", () => null);
  }

  setResponder(kind: string, responder: (msg: any) => any) {
    this.responders.set(kind, responder);
  }

  postMessage = (msg: any) => {
    this.sent.push(msg);
    const responder = this.responders.get(msg.kind);
    if (!responder) {
      return;
    }
    setTimeout(async () => {
      const content = await responder(msg);
      this.onmessage?.({ kind: "return", uuid: msg.uuid, content });
    }, 0);
  };
}

test("WebWorkerMLCEngine completion sends message to worker", async () => {
  const worker = new MockWorker();
  const engine = new WebWorkerMLCEngine(worker as any);
  await engine.reload("demo-model");
  const res = await engine.completion({
    model: "demo-model",
    prompt: "hello",
  });
  expect(res.object).toBe("completion");
  expect(worker.sent.some((msg) => msg.kind === "completionNonStreaming")).toBe(
    true,
  );
});

test("WebWorkerMLCEngine embedding delegates to worker", async () => {
  const worker = new MockWorker();
  const engine = new WebWorkerMLCEngine(worker as any);
  await engine.reload("demo-model");
  const res = await engine.embedding({
    model: "demo-model",
    input: "test",
  });
  expect(res.object).toBe("list");
  expect(worker.sent.some((msg) => msg.kind === "embedding")).toBe(true);
});

test("handleTask posts throw when task rejects", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  const postSpy = jest
    .spyOn(handler as any, "postMessage")
    .mockImplementation(() => undefined);
  await handler.handleTask("fail", async () => {
    throw new Error("boom");
  });
  expect(postSpy).toHaveBeenCalledWith(
    expect.objectContaining({
      kind: "throw",
      uuid: "fail",
    }),
  );
});

test("completionStreamNextChunk returns data from stored generator", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  const generator = (async function* () {
    yield { object: "chunk" };
  })();
  (handler as any).loadedModelIdToAsyncGenerator.set("demo", generator);
  const onComplete = jest.fn();
  handler.onmessage(
    {
      kind: "completionStreamNextChunk",
      uuid: "next",
      content: { selectedModelId: "demo" },
    } as any,
    onComplete,
  );
  await flushMicrotasks();
  expect(onComplete).toHaveBeenCalledWith({ object: "chunk" });
});

test("WebWorkerMLCEngine setAppConfig posts configuration message", () => {
  const worker = new MockWorker();
  const engine = new WebWorkerMLCEngine(worker as any);
  const config = { model_list: [] } as any;
  engine.setAppConfig(config);
  const message = worker.sent.find((msg) => msg.kind === "setAppConfig");
  expect(message).toBeDefined();
  expect(message?.content).toBe(config);
});

test("WebWorkerMLCEngine setLogLevel forwards to worker", () => {
  const worker = new MockWorker();
  const engine = new WebWorkerMLCEngine(worker as any);
  engine.setLogLevel("info" as any);
  const message = worker.sent.find((msg) => msg.kind === "setLogLevel");
  expect(message?.content).toBe("info");
});

test("WebWorkerMLCEngine info helpers resolve via worker messages", async () => {
  const worker = new MockWorker();
  worker.setResponder("getMessage", () => "ready");
  worker.setResponder("runtimeStatsText", () => "stats");
  worker.setResponder("getGPUVendor", () => "MockVendor");
  worker.setResponder("getMaxStorageBufferBindingSize", () => 2048);
  worker.setResponder("interruptGenerate", () => null);
  const engine = new WebWorkerMLCEngine(worker as any);
  await engine.reload("demo");
  await expect(engine.getMessage("demo")).resolves.toBe("ready");
  await expect(engine.runtimeStatsText()).resolves.toBe("stats");
  await expect(engine.getGPUVendor()).resolves.toBe("MockVendor");
  await expect(engine.getMaxStorageBufferBindingSize()).resolves.toBe(2048);
  engine.interruptGenerate();
  await flushMicrotasks();
  expect(worker.sent.some((msg) => msg.kind === "interruptGenerate")).toBe(
    true,
  );
});

test("Drowse worker handler routes program and measurement operations", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  const hookProgram = { enabled: Uint32Array.from([1]) } as any;
  supportsDrowseMock.mockResolvedValueOnce(true);
  setDrowseMock.mockResolvedValueOnce(undefined);
  clearDrowseMock.mockResolvedValueOnce(undefined);
  readDrowseMock.mockResolvedValueOnce(Float32Array.from([0.5]));

  handler.onmessage({
    kind: "supportsDrowseRankOneHooks",
    uuid: "supports",
    content: { modelId: "demo" },
  } as any);
  handler.onmessage({
    kind: "setDrowseRankOneProgram",
    uuid: "set",
    content: { modelId: "demo", program: hookProgram },
  } as any);
  handler.onmessage({
    kind: "clearDrowseRankOneProgram",
    uuid: "clear",
    content: { modelId: "demo" },
  } as any);
  handler.onmessage({
    kind: "readDrowseMeasurements",
    uuid: "read",
    content: { modelId: "demo" },
  } as any);
  await flushMicrotasks();

  expect(supportsDrowseMock).toHaveBeenCalledWith("demo");
  expect(setDrowseMock).toHaveBeenCalledWith(hookProgram, "demo");
  expect(clearDrowseMock).toHaveBeenCalledWith("demo");
  expect(readDrowseMock).toHaveBeenCalledWith("demo");
});

test("Drowse worker handler routes exact precomputed readout operations", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  const dictionary = {
    hookAbi: "post-block-residual-v4",
    bindingId: "a".repeat(64),
    hiddenSize: 2,
    runtimeLayerIndex: 1,
    featureCount: 2,
    encoder: Float32Array.from([1, 0, 0, 1]),
    encoderBias: new Float32Array(2),
    decoderBias: new Float32Array(2),
  };
  const jlensDictionary = {
    hookAbi: "post-block-residual-v4",
    bindingId: "b".repeat(64),
    hiddenSize: 2,
    layerIndices: Int32Array.of(1),
    matrices: [new Float32Array(4)],
  };
  const jlens = {
    tokenIds: Int32Array.from([7, 6, 5, 4, 3, 2, 1, 0]),
    strength: new Float32Array(8),
    centerOfMass: new Float32Array(8),
    spread: new Float32Array(8),
    fittedLayerCount: 1,
    layerIndices: Int32Array.of(1),
    layerTokenIds: Int32Array.from([7, 6, 5, 4, 3, 2, 1, 0]),
    layerProbabilities: new Float32Array(8),
  };
  const sae = {
    featureIds: Int32Array.from([1, 0]),
    activations: Float32Array.from([0.75, 0.25]),
    runtimeLayerIndex: 1,
    featureCount: 2,
  };
  setDrowseSaeMock.mockResolvedValueOnce(undefined);
  clearDrowseSaeMock.mockResolvedValueOnce(undefined);
  setDrowseJlensMock.mockResolvedValueOnce(undefined);
  clearDrowseJlensMock.mockResolvedValueOnce(undefined);
  resolveDrowseJlensMock.mockResolvedValueOnce(Float32Array.from([1, 2]));
  readDrowseJlensTopMock.mockResolvedValueOnce(jlens);
  readDrowseSaeTopMock.mockResolvedValueOnce(sae);
  const complete = jest.fn();

  for (const message of [
    {
      kind: "setDrowseSaeDictionary",
      uuid: "set-sae",
      content: { modelId: "demo", dictionary },
    },
    {
      kind: "clearDrowseSaeDictionary",
      uuid: "clear-sae",
      content: { modelId: "demo" },
    },
    {
      kind: "setDrowseJlensDictionary",
      uuid: "set-jlens",
      content: { modelId: "demo", dictionary: jlensDictionary },
    },
    {
      kind: "clearDrowseJlensDictionary",
      uuid: "clear-jlens",
      content: { modelId: "demo" },
    },
    {
      kind: "resolveDrowseJlensTokenDirections",
      uuid: "resolve-jlens",
      content: {
        modelId: "demo",
        bindingId: jlensDictionary.bindingId,
        layerIndices: [1],
        tokenIds: [7],
      },
    },
    {
      kind: "readDrowseJlensTopTokens",
      uuid: "read-jlens",
      content: { modelId: "demo" },
    },
    {
      kind: "readDrowseSaeTopFeatures",
      uuid: "read-sae",
      content: { modelId: "demo" },
    },
  ]) {
    handler.onmessage(message as any, complete);
  }
  await flushMicrotasks();

  expect(setDrowseSaeMock).toHaveBeenCalledWith(dictionary, "demo");
  expect(clearDrowseSaeMock).toHaveBeenCalledWith("demo");
  expect(setDrowseJlensMock).toHaveBeenCalledWith(jlensDictionary, "demo");
  expect(clearDrowseJlensMock).toHaveBeenCalledWith("demo");
  expect(resolveDrowseJlensMock).toHaveBeenCalledWith(
    jlensDictionary.bindingId,
    [1],
    [7],
    "demo",
  );
  expect(readDrowseJlensTopMock).toHaveBeenCalledWith("demo");
  expect(readDrowseSaeTopMock).toHaveBeenCalledWith("demo");
  expect(complete).toHaveBeenCalledWith(jlens);
  expect(complete).toHaveBeenCalledWith(sae);
});

test("WebWorkerMLCEngine exposes Drowse program operations", async () => {
  const worker = new MockWorker();
  const hookProgram = { enabled: Uint32Array.from([1]) } as any;
  worker.setResponder("supportsDrowseRankOneHooks", () => true);
  worker.setResponder("setDrowseRankOneProgram", () => null);
  worker.setResponder("clearDrowseRankOneProgram", () => null);
  worker.setResponder("readDrowseMeasurements", () =>
    Float32Array.from([0.25]),
  );
  const engine = new WebWorkerMLCEngine(worker as any);

  await expect(engine.supportsDrowseRankOneHooks("demo")).resolves.toBe(true);
  await engine.setDrowseRankOneProgram(hookProgram, "demo");
  await engine.clearDrowseRankOneProgram("demo");
  await expect(engine.readDrowseMeasurements("demo")).resolves.toEqual(
    Float32Array.from([0.25]),
  );
  expect(
    worker.sent.filter((message) => message.kind.includes("Drowse")),
  ).toHaveLength(4);
});

test("WebWorkerMLCEngine exposes exact precomputed readout operations", async () => {
  const worker = new MockWorker();
  const dictionary = {
    hookAbi: "post-block-residual-v4",
    bindingId: "a".repeat(64),
    hiddenSize: 2,
    runtimeLayerIndex: 1,
    featureCount: 2,
    encoder: Float32Array.from([1, 0, 0, 1]),
    encoderBias: new Float32Array(2),
    decoderBias: new Float32Array(2),
  } as any;
  const jlensDictionary = {
    hookAbi: "post-block-residual-v4",
    bindingId: "b".repeat(64),
    hiddenSize: 2,
    layerIndices: Int32Array.of(1),
    matrices: [new Float32Array(4)],
  } as any;
  const jlens = {
    tokenIds: new Int32Array(8),
    strength: new Float32Array(8),
    centerOfMass: new Float32Array(8),
    spread: new Float32Array(8),
    fittedLayerCount: 1,
    layerIndices: Int32Array.of(1),
    layerTokenIds: new Int32Array(8),
    layerProbabilities: new Float32Array(8),
  };
  const sae = {
    featureIds: Int32Array.from([1, 0]),
    activations: Float32Array.from([0.75, 0.25]),
    runtimeLayerIndex: 1,
    featureCount: 2,
  };
  worker.setResponder("setDrowseSaeDictionary", () => null);
  worker.setResponder("clearDrowseSaeDictionary", () => null);
  worker.setResponder("setDrowseJlensDictionary", () => null);
  worker.setResponder("clearDrowseJlensDictionary", () => null);
  worker.setResponder("resolveDrowseJlensTokenDirections", () =>
    Float32Array.from([1, 2]),
  );
  worker.setResponder("readDrowseJlensTopTokens", () => jlens);
  worker.setResponder("readDrowseSaeTopFeatures", () => sae);
  const engine = new WebWorkerMLCEngine(worker as any);

  await engine.setDrowseSaeDictionary(dictionary, "demo");
  await engine.setDrowseJlensDictionary(jlensDictionary, "demo");
  await expect(
    engine.resolveDrowseJlensTokenDirections(
      jlensDictionary.bindingId,
      [1],
      [7],
      "demo",
    ),
  ).resolves.toEqual(Float32Array.from([1, 2]));
  await expect(engine.readDrowseJlensTopTokens("demo")).resolves.toEqual(jlens);
  await expect(engine.readDrowseSaeTopFeatures("demo")).resolves.toEqual(sae);
  await engine.clearDrowseSaeDictionary("demo");
  await engine.clearDrowseJlensDictionary("demo");
  expect(
    worker.sent
      .filter((message) =>
        [
          "setDrowseSaeDictionary",
          "setDrowseJlensDictionary",
          "resolveDrowseJlensTokenDirections",
          "readDrowseJlensTopTokens",
          "readDrowseSaeTopFeatures",
          "clearDrowseSaeDictionary",
          "clearDrowseJlensDictionary",
        ].includes(message.kind),
      )
      .map(({ kind }) => kind),
  ).toEqual([
    "setDrowseSaeDictionary",
    "setDrowseJlensDictionary",
    "resolveDrowseJlensTokenDirections",
    "readDrowseJlensTopTokens",
    "readDrowseSaeTopFeatures",
    "clearDrowseSaeDictionary",
    "clearDrowseJlensDictionary",
  ]);
  expect(
    worker.sent.find((message) => message.kind === "setDrowseSaeDictionary")
      ?.content,
  ).toEqual({ dictionary, modelId: "demo" });
});

test("WebWorkerMLCEngine exposes Drowse capabilities and tokenizer", async () => {
  const worker = new MockWorker();
  const capabilities = {
    topK: true,
    forcedReplay: true,
    replayScoring: true,
    tokenizer: true,
    namedRoles: true,
    userSeatGeneration: true,
    sceneStitching: true,
  };
  const profile = { schemaVersion: 1, id: "standard-v1" };
  worker.setResponder("getDrowseRuntimeCapabilities", () => capabilities);
  worker.setResponder("getDrowseStructuredHookProfile", () => profile);
  worker.setResponder("tokenizeDrowseText", () => [31, 32]);
  worker.setResponder("decodeDrowseTokens", () => " leading");
  const engine = new WebWorkerMLCEngine(worker as any);

  await expect(engine.getDrowseRuntimeCapabilities("demo")).resolves.toEqual(
    capabilities,
  );
  await expect(engine.getDrowseStructuredHookProfile("demo")).resolves.toEqual(
    profile,
  );
  await expect(engine.tokenizeDrowseText(" leading", "demo")).resolves.toEqual([
    31, 32,
  ]);
  await expect(engine.decodeDrowseTokens([31, 32], "demo")).resolves.toBe(
    " leading",
  );
  expect(worker.sent.slice(-4).map((message) => message.kind)).toEqual([
    "getDrowseRuntimeCapabilities",
    "getDrowseStructuredHookProfile",
    "tokenizeDrowseText",
    "decodeDrowseTokens",
  ]);
});
