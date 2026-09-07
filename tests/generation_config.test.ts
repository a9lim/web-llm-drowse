import {
  GenerationConfig,
  postInitAndCheckGenerationConfigValues,
} from "../src/config";
import { describe, expect, test } from "@jest/globals";

describe("Check generation config illegal values", () => {
  test("Drowse deterministic sampler boundary values are valid", () => {
    expect(() =>
      postInitAndCheckGenerationConfigValues({
        temperature: -1,
        top_p: 0,
        top_k: 0,
      }),
    ).not.toThrow();
  });

  test("Drowse exact sampling fields reject invalid values", () => {
    for (const top_k of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => postInitAndCheckGenerationConfigValues({ top_k })).toThrow();
    }
    for (const temperature of [
      Number.NaN,
      Number.POSITIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
    ]) {
      expect(() =>
        postInitAndCheckGenerationConfigValues({ temperature }),
      ).toThrow("temperature must be finite");
    }
    expect(() =>
      postInitAndCheckGenerationConfigValues({
        drowse_generation_seat: "tool" as any,
      }),
    ).toThrow();
    for (const tokenIds of [[-1], [1.5], [Number.MAX_SAFE_INTEGER + 1]]) {
      expect(() =>
        postInitAndCheckGenerationConfigValues({
          drowse_forced_prefix_token_ids: tokenIds,
        }),
      ).toThrow();
    }
  });
  test("High-level unsupported fields", () => {
    expect(() => {
      const genConfig: GenerationConfig = {
        max_tokens: 0,
      };
      postInitAndCheckGenerationConfigValues(genConfig);
    }).toThrow("Make sure `max_tokens` > 0");
  });

  test("logit_bias exceeds range", () => {
    expect(() => {
      const genConfig: GenerationConfig = {
        max_tokens: 10,
        logit_bias: {
          "1355": 155,
        },
      };
      postInitAndCheckGenerationConfigValues(genConfig);
    }).toThrow("Make sure -100 < logit_bias <= 100.");
  });

  test("logit_bias invalid key", () => {
    expect(() => {
      const genConfig: GenerationConfig = {
        max_tokens: 10,
        logit_bias: {
          thisRaisesError: 50,
        },
      };
      postInitAndCheckGenerationConfigValues(genConfig);
    }).toThrow(
      "Make sure logit_bias's keys to be number represented in string.",
    );
  });

  test("top_logprobs out of range", () => {
    expect(() => {
      const genConfig: GenerationConfig = {
        logprobs: true,
        top_logprobs: 6,
        max_tokens: 10,
      };
      postInitAndCheckGenerationConfigValues(genConfig);
    }).toThrow("Make sure 0 < top_logprobs <= 5.");
  });

  test("top_logprobs set without setting logprobs", () => {
    expect(() => {
      const genConfig: GenerationConfig = {
        top_logprobs: 3,
        max_tokens: 10,
      };
      postInitAndCheckGenerationConfigValues(genConfig);
    }).toThrow("top_logprobs requires logprobs to be true");
  });

  test("top_logprobs set though logprobs is false", () => {
    expect(() => {
      const genConfig: GenerationConfig = {
        logprobs: false,
        top_logprobs: 3,
        max_tokens: 10,
      };
      postInitAndCheckGenerationConfigValues(genConfig);
    }).toThrow("top_logprobs requires logprobs to be true");
  });
});

describe("Check generation post init", () => {
  test("Drowse exact sampling fields accept zero top-k and token IDs", () => {
    const genConfig: GenerationConfig = {
      top_k: 0,
      drowse_generation_seat: "user",
      drowse_forced_prefix_token_ids: [0, 17],
      drowse_score_token_ids: [17, 23],
    };
    postInitAndCheckGenerationConfigValues(genConfig);
    expect(genConfig).toEqual({
      top_k: 0,
      drowse_generation_seat: "user",
      drowse_forced_prefix_token_ids: [0, 17],
      drowse_score_token_ids: [17, 23],
    });
  });
  test("Only set one of presence or frequency penalty", () => {
    const genConfig: GenerationConfig = {
      frequency_penalty: 1.5,
    };
    postInitAndCheckGenerationConfigValues(genConfig);
    expect(genConfig.presence_penalty).toBe(0.0);
  });

  test("Set logprobs without setting top_logprobs", () => {
    const genConfig: GenerationConfig = {
      logprobs: true,
    };
    postInitAndCheckGenerationConfigValues(genConfig);
    expect(genConfig.top_logprobs).toBe(0);
  });

  test("Set both logprobs and top_logprobs", () => {
    const genConfig: GenerationConfig = {
      logprobs: true,
      top_logprobs: 2,
    };
    postInitAndCheckGenerationConfigValues(genConfig);
    expect(genConfig.top_logprobs).toBe(2);
  });
});

describe("Reject NaN generation values", () => {
  const nanCases: Array<[string, GenerationConfig, string]> = [
    [
      "frequency_penalty",
      { frequency_penalty: Number.NaN },
      "Make sure -2 < frequency_penalty <= 2.",
    ],
    [
      "presence_penalty",
      { presence_penalty: Number.NaN },
      "Make sure -2 < presence_penalty <= 2.",
    ],
    [
      "repetition_penalty",
      { repetition_penalty: Number.NaN },
      "Make sure `repetition_penalty` > 0.",
    ],
    ["max_tokens", { max_tokens: Number.NaN }, "Make sure `max_tokens` > 0."],
    ["top_p", { top_p: Number.NaN }, "Make sure 0 < top_p <= 1."],
    ["temperature", { temperature: Number.NaN }, "temperature must be finite"],
    [
      "top_logprobs",
      { logprobs: true, top_logprobs: Number.NaN },
      "Make sure 0 < top_logprobs <= 5.",
    ],
    [
      "logit_bias value",
      { logit_bias: { "1": Number.NaN } },
      "Make sure -100 < logit_bias <= 100.",
    ],
  ];

  test.each(nanCases)("rejects NaN in %s", (_name, config, message) => {
    expect(() => postInitAndCheckGenerationConfigValues(config)).toThrow(
      message,
    );
  });
});

describe("Preserve generation value semantics", () => {
  test("accepts null, undefined, zero, and positive infinity where supported", () => {
    const config: GenerationConfig = {
      repetition_penalty: Number.POSITIVE_INFINITY,
      temperature: 1,
      max_tokens: Number.POSITIVE_INFINITY,
      frequency_penalty: 0,
      presence_penalty: 0,
      top_logprobs: 0,
      logprobs: true,
      top_p: null,
      logit_bias: null,
    };

    expect(() => postInitAndCheckGenerationConfigValues(config)).not.toThrow();
    expect(config.temperature).toBe(1);
    expect(config.top_p).toBeNull();
  });

  test("accepts inclusive range endpoints", () => {
    const config: GenerationConfig = {
      frequency_penalty: -2,
      presence_penalty: 2,
      top_p: 1,
      temperature: 0,
      repetition_penalty: Number.MIN_VALUE,
      max_tokens: 1,
      logprobs: true,
      top_logprobs: 5,
      logit_bias: { "1": -100, "2": 100 },
    };

    expect(() => postInitAndCheckGenerationConfigValues(config)).not.toThrow();
  });

  test("keeps undefined generation values omitted", () => {
    const config: GenerationConfig = {
      repetition_penalty: undefined,
      temperature: undefined,
      max_tokens: undefined,
      frequency_penalty: undefined,
      presence_penalty: undefined,
      top_logprobs: undefined,
      logit_bias: undefined,
    };

    postInitAndCheckGenerationConfigValues(config);
    expect(config).toEqual({
      repetition_penalty: undefined,
      temperature: undefined,
      max_tokens: undefined,
      frequency_penalty: undefined,
      presence_penalty: undefined,
      top_logprobs: undefined,
      logit_bias: undefined,
    });
  });
});
