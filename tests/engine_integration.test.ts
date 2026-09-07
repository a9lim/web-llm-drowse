/**
 * Deterministic MLCEngine tests that run without WebGPU by mocking LLMChatPipeline.
 */
import {
  ChatCompletion,
  ChatCompletionRequest,
  Completion,
  CompletionCreateParams,
  EmbeddingCreateParams,
  ChatCompletionChunk,
} from "../src/openai_api_protocols";
import { MLCEngine } from "../src/engine";
import { ModelType } from "../src/config";
import { LLMChatPipeline } from "../src/llm_chat";
import { EmbeddingPipeline } from "../src/embedding";
import { CustomLock } from "../src/support";
import { UnclearModelToUseError } from "../src/error";
import { jest, test, expect, describe, afterEach } from "@jest/globals";

type ChatConfig = import("../src/config").ChatConfig;
type Conversation = import("../src/conversation").Conversation;
type TVMInstance = import("@mlc-ai/web-runtime").Instance;
type Tokenizer = import("@mlc-ai/web-tokenizers").Tokenizer;

jest.mock("../src/llm_chat", () => {
  const { getConversation } = jest.requireActual(
    "../src/conversation",
  ) as typeof import("../src/conversation");

  class MockLLMChatPipeline {
    public decodeLimit = 2;
    public prefillCallCount = 0;
    public decodeCallCount = 0;
    public resetCount = 0;
    public lastGenConfig: Record<string, unknown> | undefined;
    private conversation: Conversation = getConversation(
      {
        system_template: "{system_message}",
        system_message: "",
        roles: { user: "user", assistant: "assistant" },
        seps: ["\n"],
        stop_token_ids: [0],
        stop_str: [],
      } as any,
      undefined,
    );
    private stopFlag = true;
    private message = "";
    private finishReason: string | undefined = undefined;
    private drowseFinishReason:
      | "eos"
      | "stop_sequence"
      | "external_stop"
      | "length"
      | undefined = undefined;
    private curRoundPrefillTotalTokens = 0;
    private curRoundDecodingTotalTokens = 0;
    private curRoundPrefillTotalTime = 0.001;
    private curRoundDecodingTotalTime = 0.001;
    private curRoundGrammarPerTokenTotalTime = 0;

    constructor(_tvm: TVMInstance, _tokenizer: Tokenizer, config: ChatConfig) {
      this.conversation = getConversation(
        config.conv_template,
        config.conv_config,
      );
    }

    async asyncLoadWebGPUPipelines() {}
    dispose() {}
    async sync() {}

    getConversationObject() {
      return this.conversation;
    }

    setConversation(newConv: Conversation) {
      this.conversation = newConv;
    }

    resetChat() {
      this.resetCount++;
      this.stopFlag = true;
      this.decodeCallCount = 0;
    }

    async prefillStep(
      inp: string,
      msgRole: string,
      roleName?: string,
      genConfig?: Record<string, unknown>,
    ): Promise<void> {
      this.lastGenConfig = genConfig;
      this.prefillCallCount++;
      const roleSuffix = roleName ? `(${roleName})` : "";
      this.message = `${msgRole}${roleSuffix}:${inp}`;
      this.stopFlag = false;
      this.decodeCallCount = 0;
      this.curRoundPrefillTotalTokens = Math.max(1, inp.length);
      this.curRoundPrefillTotalTime = 0.01 * this.curRoundPrefillTotalTokens;
      this.curRoundDecodingTotalTokens = 0;
      this.curRoundDecodingTotalTime = 0.001;
      this.curRoundGrammarPerTokenTotalTime = 0;
      this.finishReason = "length";
      this.drowseFinishReason = "length";
    }

    async decodeStep(genConfig?: { max_tokens?: number | null }) {
      if (this.stopFlag) return;
      this.decodeCallCount++;
      this.message += `|token${this.decodeCallCount}|`;
      this.curRoundDecodingTotalTokens = this.decodeCallCount;
      this.curRoundDecodingTotalTime = this.curRoundDecodingTotalTokens * 0.02;
      this.curRoundGrammarPerTokenTotalTime =
        this.curRoundDecodingTotalTokens * 0.001;
      if (
        this.decodeCallCount >= this.decodeLimit ||
        (genConfig?.max_tokens !== null &&
          genConfig?.max_tokens !== undefined &&
          this.decodeCallCount >= genConfig.max_tokens)
      ) {
        this.stopFlag = true;
        this.finishReason = "stop";
        this.drowseFinishReason = "eos";
      }
    }

    stopped() {
      return this.stopFlag;
    }

    triggerStop() {
      this.stopFlag = true;
      this.finishReason = "abort";
      this.drowseFinishReason = "external_stop";
    }

    getMessage() {
      return this.message;
    }

    getDrowseRawMessage() {
      return this.getMessage();
    }

    getFinishReason() {
      return this.finishReason ?? "stop";
    }

    getDrowseFinishReason() {
      return this.drowseFinishReason;
    }

    getCurRoundDecodingTotalTokens() {
      return this.curRoundDecodingTotalTokens;
    }

    getCurRoundDrowseCompletionTokens() {
      return this.curRoundDecodingTotalTokens;
    }

    getCurRoundPrefillTotalTokens() {
      return this.curRoundPrefillTotalTokens;
    }

    getCurRoundPrefillTokensPerSec() {
      return this.curRoundPrefillTotalTokens / this.curRoundPrefillTotalTime;
    }

    getCurRoundDecodingTokensPerSec() {
      return this.curRoundDecodingTotalTokens / this.curRoundDecodingTotalTime;
    }

    getCurRoundGrammarInitTotalTime() {
      return 0.001;
    }

    getCurRoundPrefillTotalTime() {
      return this.curRoundPrefillTotalTime;
    }

    getCurRoundDecodingTotalTime() {
      return this.curRoundDecodingTotalTime;
    }

    getCurRoundGrammarPerTokenTotalTime() {
      return this.curRoundGrammarPerTokenTotalTime;
    }

    getCurRoundLatencyBreakdown() {
      return {
        logitProcessorTime: [0.001],
        logitBiasTime: [0.001],
        penaltyTime: [0.001],
        sampleTime: [0.001],
        totalTime: [0.001],
        grammarBitmaskTime: [0.001],
      };
    }

    getTokenLogprobArray() {
      return [];
    }

    async forwardTokensAndSample(inputIds: Array<number>): Promise<number> {
      return inputIds[0] ?? 0;
    }

    getDrowseRuntimeCapabilities() {
      return {
        topK: true,
        forcedReplay: true,
        replayScoring: true,
        tokenizer: true,
        namedRoles: false,
        userSeatGeneration: true,
        sceneStitching: true,
      };
    }

    tokenizeDrowseText(text: string) {
      return text === " leading" ? [31, 32] : [41, 42, 43];
    }

    decodeDrowseTokens(tokenIds: readonly number[]) {
      return tokenIds.join(":");
    }

    async setDrowseStructuredProgram() {}

    async runtimeStatsText() {
      return `prefills=${this.prefillCallCount}`;
    }
  }

  return { LLMChatPipeline: MockLLMChatPipeline };
});

jest.mock("../src/embedding", () => {
  class MockEmbeddingPipeline {
    public inputs: any;
    public embedResult: Array<Array<number>> = [[0.1, 0.2, 0.3]];
    dispose() {}
    async sync() {}
    async embedStep(
      input: string | Array<string> | Array<number> | Array<Array<number>>,
    ): Promise<Array<Array<number>>> {
      this.inputs = input;
      return this.embedResult;
    }
    getCurRoundEmbedTotalTokens(): number {
      if (typeof this.inputs === "string") {
        return this.inputs.length;
      } else if (Array.isArray(this.inputs)) {
        return this.inputs.length;
      }
      return 0;
    }
    getCurRoundEmbedTokensPerSec(): number {
      const tokens = this.getCurRoundEmbedTotalTokens();
      return tokens === 0 ? 0 : tokens / 0.01;
    }
  }
  return { EmbeddingPipeline: MockEmbeddingPipeline };
});

const MODEL_ID = "mock-model";
const SECOND_MODEL_ID = "mock-model-2";
const EMBED_MODEL_ID = "mock-embed";
const FIXED_CREATED_DATE = new Date("2024-04-05T06:34:56.789Z");
const FIXED_CREATED_SECONDS = 1712298896;
const mockChatConfig: ChatConfig = {
  tokenizer_files: ["tokenizer.json"],
  vocab_size: 10,
  conv_template: {
    system_template: "{system_message}",
    system_message: "You are a helpful assistant.",
    system_prefix_token_ids: [1],
    add_role_after_system_message: false,
    roles: {
      user: "User",
      assistant: "Assistant",
      tool: "Tool",
    },
    role_templates: {
      user: "{user_message}",
      assistant: "{assistant_message}",
      tool: "{tool_message}",
    },
    seps: ["\n"],
    role_content_sep: ": ",
    role_empty_sep: ": ",
    stop_str: [],
    stop_token_ids: [0],
  },
  conv_config: undefined,
  context_window_size: 8,
  sliding_window_size: -1,
  attention_sink_size: -1,
  temperature: 0.8,
  presence_penalty: 0,
  frequency_penalty: 0,
  repetition_penalty: 1,
  top_p: 1,
};

function createEngineWithPipeline(decodeLimit = 2, modelId = MODEL_ID) {
  const engine = new MLCEngine({
    appConfig: {
      model_list: [
        {
          model: "https://example.com/model",
          model_id: modelId,
          model_lib: "https://example.com/model.wasm",
        },
      ],
      cacheBackend: "cache",
    },
  });
  const pipeline = new LLMChatPipeline(
    null as unknown as TVMInstance,
    null as unknown as Tokenizer,
    mockChatConfig,
  ) as any;
  pipeline.decodeLimit = decodeLimit;
  const internal = engine as any;
  internal.loadedModelIdToPipeline.set(modelId, pipeline);
  internal.loadedModelIdToChatConfig.set(modelId, mockChatConfig);
  internal.loadedModelIdToModelType.set(modelId, ModelType.LLM);
  internal.loadedModelIdToLock.set(modelId, new CustomLock());
  return { engine, pipeline };
}

function createEngineWithMultiplePipelines() {
  const engine = new MLCEngine({
    appConfig: {
      model_list: [
        {
          model: "https://example.com/model",
          model_id: MODEL_ID,
          model_lib: "https://example.com/model.wasm",
        },
        {
          model: "https://example.com/model2",
          model_id: SECOND_MODEL_ID,
          model_lib: "https://example.com/model2.wasm",
        },
      ],
      cacheBackend: "cache",
    },
  });
  const pipeline1 = new LLMChatPipeline(
    null as unknown as TVMInstance,
    null as unknown as Tokenizer,
    mockChatConfig,
  ) as any;
  const pipeline2 = new LLMChatPipeline(
    null as unknown as TVMInstance,
    null as unknown as Tokenizer,
    mockChatConfig,
  ) as any;
  const internal = engine as any;
  internal.loadedModelIdToPipeline.set(MODEL_ID, pipeline1);
  internal.loadedModelIdToPipeline.set(SECOND_MODEL_ID, pipeline2);
  internal.loadedModelIdToChatConfig.set(MODEL_ID, mockChatConfig);
  internal.loadedModelIdToChatConfig.set(SECOND_MODEL_ID, mockChatConfig);
  internal.loadedModelIdToModelType.set(MODEL_ID, ModelType.LLM);
  internal.loadedModelIdToModelType.set(SECOND_MODEL_ID, ModelType.LLM);
  internal.loadedModelIdToLock.set(MODEL_ID, new CustomLock());
  internal.loadedModelIdToLock.set(SECOND_MODEL_ID, new CustomLock());
  return engine;
}

const mockEmbeddingConfig: ChatConfig = {
  ...mockChatConfig,
};

function createEngineWithEmbeddingPipeline() {
  const engine = new MLCEngine({
    appConfig: {
      model_list: [
        {
          model: "https://example.com/embed",
          model_id: EMBED_MODEL_ID,
          model_lib: "https://example.com/embed.wasm",
          model_type: ModelType.embedding,
        },
      ],
      cacheBackend: "cache",
    },
  });
  const pipeline = new EmbeddingPipeline(
    null as unknown as TVMInstance,
    null as unknown as Tokenizer,
    mockEmbeddingConfig,
  ) as any;
  const internal = engine as any;
  internal.loadedModelIdToPipeline.set(EMBED_MODEL_ID, pipeline);
  internal.loadedModelIdToChatConfig.set(EMBED_MODEL_ID, mockEmbeddingConfig);
  internal.loadedModelIdToModelType.set(EMBED_MODEL_ID, ModelType.embedding);
  internal.loadedModelIdToLock.set(EMBED_MODEL_ID, new CustomLock());
  return { engine, pipeline };
}

afterEach(() => {
  jest.useRealTimers();
});

describe("MLCEngine deterministic integration", () => {
  test("chatCompletion aggregates usage without WebGPU", async () => {
    jest.useFakeTimers().setSystemTime(FIXED_CREATED_DATE);
    const { engine, pipeline } = createEngineWithPipeline(3);
    const request: ChatCompletionRequest = {
      model: MODEL_ID,
      messages: [
        { role: "system", content: "Stay concise." },
        { role: "user", content: "What is new?" },
      ],
      n: 2,
    };
    const response = (await engine.chatCompletion(request)) as ChatCompletion;

    expect(response.choices).toHaveLength(2);
    response.choices.forEach((choice) => {
      expect(choice.message?.content).toContain("What is new?");
    });
    expect(response.created).toBe(FIXED_CREATED_SECONDS);
    expect(response.usage?.completion_tokens).toBe(6);
    expect(response.usage?.prompt_tokens).toBeGreaterThan(0);
    expect((pipeline as any).prefillCallCount).toBe(2);
  });

  test("completion echoes prompt when requested", async () => {
    jest.useFakeTimers().setSystemTime(FIXED_CREATED_DATE);
    const { engine } = createEngineWithPipeline(1);
    const request: CompletionCreateParams = {
      model: MODEL_ID,
      prompt: "Alpha ",
      n: 1,
      echo: true,
    };
    const response = (await engine.completion(request)) as Completion;

    expect(response.choices).toHaveLength(1);
    expect(response.choices[0].text.startsWith("Alpha ")).toBe(true);
    expect(response.created).toBe(FIXED_CREATED_SECONDS);
    expect(response.usage?.completion_tokens).toBe(1);
    expect(response.usage?.prompt_tokens).toBeGreaterThan(0);
  });

  test("forwardTokensAndSample and runtimeStatsText use mock pipeline", async () => {
    const { engine } = createEngineWithPipeline();
    await expect(
      engine.forwardTokensAndSample([9, 4, 2], true, MODEL_ID),
    ).resolves.toBe(9);
    await expect(engine.runtimeStatsText(MODEL_ID)).resolves.toContain(
      "prefills=",
    );
  });

  test("Drowse capabilities and tokenizer bridge use the selected pipeline", async () => {
    const { engine } = createEngineWithPipeline();
    await expect(
      engine.getDrowseRuntimeCapabilities(MODEL_ID),
    ).resolves.toEqual({
      topK: true,
      forcedReplay: true,
      replayScoring: true,
      tokenizer: true,
      namedRoles: false,
      userSeatGeneration: true,
      sceneStitching: true,
    });
    await expect(
      engine.tokenizeDrowseText(" leading", MODEL_ID),
    ).resolves.toEqual([31, 32]);
    await expect(
      engine.tokenizeDrowseText("subword", MODEL_ID),
    ).resolves.toEqual([41, 42, 43]);
    await expect(engine.decodeDrowseTokens([31, 32], MODEL_ID)).resolves.toBe(
      "31:32",
    );
  });

  test("Drowse structured program installation waits for GPU uploads", async () => {
    const { engine, pipeline } = createEngineWithPipeline();
    let finishUpload!: () => void;
    const upload = new Promise<void>((resolve) => {
      finishUpload = resolve;
    });
    pipeline.setDrowseStructuredProgram = jest.fn(() => upload);
    let installed = false;
    const installing = engine
      .setDrowseStructuredProgram({} as any, MODEL_ID)
      .then(() => {
        installed = true;
      });

    await Promise.resolve();
    expect(installed).toBe(false);
    finishUpload();
    await installing;
    expect(installed).toBe(true);
  });

  test("chatCompletion forwards exact Drowse sampling extensions", async () => {
    const { engine, pipeline } = createEngineWithPipeline(1);
    await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "score" }],
      top_k: 17,
      extra_body: {
        drowse_forced_prefix_token_ids: [4, 5],
        drowse_score_token_ids: [4, 9],
      },
    });
    expect((pipeline as any).lastGenConfig).toMatchObject({
      top_k: 17,
      drowse_forced_prefix_token_ids: [4, 5],
      drowse_score_token_ids: [4, 9],
    });
  });

  test("chatCompletion streaming yields chunks, final delta, and usage data", async () => {
    jest.useFakeTimers().setSystemTime(FIXED_CREATED_DATE);
    const { engine } = createEngineWithPipeline(2);
    const request: ChatCompletionRequest = {
      model: MODEL_ID,
      messages: [
        { role: "system", content: "rules" },
        { role: "user", content: "Stream please" },
      ],
      stream: true,
      stream_options: { include_usage: true },
    };
    const iterable = (await engine.chatCompletion(
      request,
    )) as AsyncIterable<ChatCompletionChunk>;
    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of iterable) {
      chunks.push(chunk);
    }
    expect(chunks.length).toBeGreaterThanOrEqual(3);
    expect(chunks[0].choices[0].delta?.content).toContain("Stream please");
    expect(
      chunks.every((chunk) => chunk.created === FIXED_CREATED_SECONDS),
    ).toBe(true);
    const finalChunk = chunks[chunks.length - 2];
    expect(finalChunk.choices[0].finish_reason).toEqual("stop");
    const usageChunk = chunks[chunks.length - 1];
    expect(usageChunk.usage?.completion_tokens).toBeGreaterThan(0);
    expect(usageChunk.usage?.prompt_tokens).toBeGreaterThan(0);
  });

  test("streaming preserves every token boundary across split UTF-8 text", async () => {
    const { engine, pipeline } = createEngineWithPipeline(2);
    let step = 0;
    const rows = [
      {
        token_id: 100,
        token: "�",
        bytes: [0xef, 0xbf, 0xbd],
        logprob: -0.2,
        top_logprobs: [],
      },
      {
        token_id: 101,
        token: "�",
        bytes: [0xef, 0xbf, 0xbd],
        logprob: -0.3,
        top_logprobs: [],
      },
    ];
    pipeline.prefillStep = jest.fn(async () => {
      step = 1;
    });
    pipeline.decodeStep = jest.fn(async () => {
      step = 2;
    });
    pipeline.stopped = jest.fn(() => step >= 2);
    pipeline.getMessage = jest.fn(() => (step === 1 ? "�" : "😀"));
    pipeline.getTokenLogprobArray = jest.fn(() => rows.slice(0, step));
    pipeline.getFinishReason = jest.fn(() => "stop");
    pipeline.getDrowseFinishReason = jest.fn(() => "eos");
    pipeline.getCurRoundDrowseCompletionTokens = jest.fn(() => 2);

    const iterable = (await engine.completion({
      model: MODEL_ID,
      prompt: "emoji",
      max_tokens: 2,
      stream: true,
      logprobs: true,
      top_logprobs: 1,
    })) as AsyncIterable<Completion>;
    const chunks: Completion[] = [];
    for await (const chunk of iterable) chunks.push(chunk);
    const tokenChunks = chunks.filter(
      (chunk) => chunk.choices[0].logprobs?.content?.length === 1,
    );

    expect(tokenChunks).toHaveLength(2);
    expect(tokenChunks.map((chunk) => chunk.choices[0].text)).toEqual([
      "",
      "😀",
    ]);
    expect(
      tokenChunks.map((chunk) => chunk.choices[0].logprobs?.content?.[0]),
    ).toMatchObject([
      { token_id: 100, token: "", bytes: [] },
      { token_id: 101, token: "😀", bytes: [240, 159, 152, 128] },
    ]);
  });

  test("Drowse token decoding does not re-enter the model lock while a stream chunk is yielded", async () => {
    const { engine } = createEngineWithPipeline(1);
    const stream = (await engine.completion({
      model: MODEL_ID,
      prompt: "decode this chunk",
      max_tokens: 1,
      stream: true,
    })) as AsyncIterable<Completion>;
    const lock = (engine as any).loadedModelIdToLock.get(MODEL_ID);

    for await (const chunk of stream) {
      expect(chunk.object).toBe("text_completion");
      const acquire = jest
        .spyOn(lock, "acquire")
        .mockRejectedValue(new Error("stream lock re-entry"));
      await expect(engine.decodeDrowseTokens([31, 32], MODEL_ID)).resolves.toBe(
        "31:32",
      );
      expect(acquire).not.toHaveBeenCalled();
      acquire.mockRestore();
      break;
    }
  });

  test("a completed interrupt does not cancel the next request", async () => {
    const { engine, pipeline } = createEngineWithPipeline(8);
    const stream = (await engine.completion({
      model: MODEL_ID,
      prompt: "interrupt me",
      max_tokens: 8,
      stream: true,
    })) as AsyncIterable<Completion>;
    let chunks = 0;
    for await (const chunk of stream) {
      expect(chunk.object).toBe("text_completion");
      chunks += 1;
      if (chunks === 1) await engine.interruptGenerate();
    }
    const prefillCount = (pipeline as any).prefillCallCount;
    const response = (await engine.completion({
      model: MODEL_ID,
      prompt: "run next",
      max_tokens: 1,
    })) as Completion;
    expect((pipeline as any).prefillCallCount).toBe(prefillCount + 1);
    expect(response.choices[0].text).toContain("run next");
  });

  test("chatCompletion without specifying model when multiple loaded throws error", async () => {
    const engine = createEngineWithMultiplePipelines();
    await expect(
      engine.chatCompletion({
        // purposely omit model to trigger ambiguity
        model: undefined as any,
        messages: [{ role: "user", content: "Hello" }],
      }),
    ).rejects.toBeInstanceOf(UnclearModelToUseError);
  });

  test("embedding API uses mock pipeline and returns usage", async () => {
    const { engine } = createEngineWithEmbeddingPipeline();
    const request: EmbeddingCreateParams = {
      model: EMBED_MODEL_ID,
      input: "abc",
    };
    const response = await engine.embedding(request);
    expect(response.data).toHaveLength(1);
    expect(response.data[0].embedding).toEqual([0.1, 0.2, 0.3]);
    expect(response.usage?.prompt_tokens).toBeGreaterThan(0);
    expect(response.usage?.extra?.prefill_tokens_per_s).toBeGreaterThan(0);
  });
});
