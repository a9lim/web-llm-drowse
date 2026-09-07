import { ChatConfig, Role } from "../src/config";
import {
  compareConversationObject,
  getConversation,
  getConversationFromChatCompletionRequest,
} from "../src/conversation";
import { describe, expect, test } from "@jest/globals";
import {
  llama2ChatConfigJSONString,
  phi3_5VisionChatConfigJSONString,
  qwen3ChatConfigJSONString,
} from "./constants";
import {
  ChatCompletionContentPartImage,
  ChatCompletionMessageParam,
  ChatCompletionRequest,
} from "../src/openai_api_protocols";

describe("Test basic conversation loading and getPromptArray", () => {
  test("Test from json", () => {
    const config_json = JSON.parse(llama2ChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    const conversation = getConversation(config.conv_template);
    const config_obj = conversation.config;

    expect(config_obj.system_template).toEqual(
      "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
    );
    expect(config_obj.system_message).toEqual(
      "You are a helpful, respectful and honest assistant.",
    );
    expect(config_obj.roles.user).toEqual("[INST]");
    expect(config_obj.roles.assistant).toEqual("[/INST]");
    expect(config_obj.role_templates?.user).toEqual("{user_message}");
    expect(config_obj.role_templates?.assistant).toEqual("{assistant_message}");
    expect(config_obj.role_content_sep).toEqual(" ");
    expect(config_obj.role_empty_sep).toEqual(" ");
    expect(config_obj.seps).toEqual([" "]);
    expect(config_obj.stop_str).toEqual(["[INST]"]);
    expect(config_obj.stop_token_ids).toEqual([2]);
    expect(config_obj.system_prefix_token_ids).toEqual([1]);
    expect(config_obj.add_role_after_system_message).toBe(false);

    conversation.appendMessage(Role.user, "test1");
    conversation.appendMessage(Role.assistant, "test2");
    conversation.appendMessage(Role.user, "test3");
    conversation.appendReplyHeader(Role.assistant);
    const prompt = conversation.getPromptArray().join("");
    expect(prompt).toEqual(
      "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\ntest1 [/INST] test2 [INST] test3 [/INST] ",
    );
  });
});

describe("Test getConversationFromChatCompletionRequest with Qwen3", () => {
  test("renders named user and assistant history in their original structural seats", () => {
    const config_json = JSON.parse(qwen3ChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    const conversation = getConversationFromChatCompletionRequest(
      {
        messages: [
          { role: "user", content: "hello", name: "curious_user" },
          { role: "assistant", content: "welcome", name: "prior_guide" },
          { role: "user", content: "continue" },
        ],
      },
      config,
      true,
    );

    expect(conversation.getPromptArray().join("")).toEqual(
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
        "<|im_start|>curious user\nhello<|im_end|>\n" +
        "<|im_start|>prior guide\nwelcome<|im_end|>\n" +
        "<|im_start|>user\ncontinue<|im_end|>\n",
    );
    expect(conversation.messages.map((message) => message[0])).toEqual([
      Role.user,
      Role.assistant,
      Role.user,
    ]);
  });

  test("renders a Drowse assistant reply role without changing the assistant seat", () => {
    const config_json = JSON.parse(qwen3ChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    const conversation = getConversation(config.conv_template);

    conversation.appendMessage(Role.user, "hello");
    conversation.appendReplyHeader(Role.assistant, "someone_happy");

    expect(conversation.getPromptArray().join("")).toEqual(
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
        "<|im_start|>user\nhello<|im_end|>\n" +
        "<|im_start|>someone happy\n",
    );
    expect(conversation.messages.at(-1)?.[0]).toBe(Role.assistant);
    expect(conversation.supportsDrowseNamedRoles()).toBe(true);
    expect(conversation.supportsDrowseUserSeatGeneration()).toBe(true);
  });

  test("renders a named Qwen user-seat reply after assistant history", () => {
    const config_json = JSON.parse(qwen3ChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    const conversation = getConversationFromChatCompletionRequest(
      {
        messages: [{ role: "assistant", content: "Where should we go?" }],
        extra_body: { drowse_generation_seat: "user" },
      },
      config,
      true,
    );

    conversation.appendReplyHeader(Role.user, "curious_user");
    expect(conversation.getPromptArray().join("")).toContain(
      "<|im_start|>assistant\nWhere should we go?<|im_end|>\n" +
        "<|im_start|>curious user\n",
    );
    expect(conversation.messages.at(-1)?.[0]).toBe(Role.user);
  });

  test("Test Qwen3 appendEmptyThinkingReplyHeader", () => {
    const config_json = JSON.parse(qwen3ChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    const conversation = getConversation(config.conv_template);

    conversation.appendMessage(Role.user, "test1");
    conversation.appendMessage(Role.assistant, "test2");
    const emptyThinkingBlockStr = "<think>\n\n</think>\n\n";
    conversation.appendEmptyThinkingReplyHeader(
      Role.user,
      emptyThinkingBlockStr,
    );
    const prompt = conversation.getPromptArray().join("");
    expect(prompt).toEqual(
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
        "<|im_start|>user\n" +
        "test1<|im_end|>\n" +
        "<|im_start|>assistant\n" +
        "test2<|im_end|>\n" +
        "<|im_start|>user\n" +
        emptyThinkingBlockStr,
    );

    const message = emptyThinkingBlockStr + "test3";
    conversation.finishReply(message);
    expect(conversation.messages[conversation.messages.length - 1][2]).toEqual(
      message,
    );
  });
});

describe("Test SmolLM2 Drowse reply role rendering", () => {
  test("preserves SmolLM2 ChatML markers and de-slugs the generated label", () => {
    const conversation = getConversation({
      system_template: "<|im_start|>system\n{system_message}<|im_end|>\n",
      system_message:
        "You are a helpful AI assistant named SmolLM, trained by Hugging Face",
      roles: {
        user: "<|im_start|>user",
        assistant: "<|im_start|>assistant",
        tool: "<|im_start|>tool",
      },
      role_templates: {
        user: "{user_message}",
        assistant: "{assistant_message}",
      },
      role_content_sep: "\n",
      role_empty_sep: "\n",
      seps: ["<|im_end|>\n"],
      stop_str: ["<|im_end|>"],
      stop_token_ids: [2],
      add_role_after_system_message: true,
    });

    conversation.appendMessage(Role.user, "hello");
    conversation.appendReplyHeader(Role.assistant, "forest_guide");

    expect(conversation.getPromptArray().join("")).toEqual(
      "<|im_start|>system\n" +
        "You are a helpful AI assistant named SmolLM, trained by Hugging Face" +
        "<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n" +
        "<|im_start|>forest guide\n",
    );
    expect(conversation.messages.at(-1)?.[0]).toBe(Role.assistant);

    conversation.finishReply("welcome");
    conversation.appendReplyHeader(Role.user, "curious_user");
    expect(conversation.getPromptArray().join("")).toContain(
      "<|im_start|>forest guide\nwelcome<|im_end|>\n" +
        "<|im_start|>curious user\n",
    );
  });

  test("preserves ordered system turns for activation capture", () => {
    const conversation = getConversation({
      system_template: "<|im_start|>system\n{system_message}<|im_end|>\n",
      system_message: "default",
      roles: {
        user: "<|im_start|>user",
        assistant: "<|im_start|>assistant",
        tool: "<|im_start|>tool",
      },
      role_content_sep: "\n",
      role_empty_sep: "\n",
      seps: ["<|im_end|>\n"],
      stop_str: ["<|im_end|>"],
      stop_token_ids: [2],
    });
    conversation.override_system_message = "directive";
    conversation.appendMessage(Role.user, "first");
    conversation.appendDrowseSystemMessage("second system");
    conversation.appendMessage(Role.assistant, "reply");

    expect(conversation.getPromptArray().join("")).toEqual(
      "<|im_start|>system\ndirective<|im_end|>\n" +
        "<|im_start|>user\nfirst<|im_end|>\n" +
        "<|im_start|>system\nsecond system<|im_end|>\n" +
        "<|im_start|>assistant\nreply<|im_end|>\n",
    );
  });

  test("rejects ordered system turns for templates without an exact system segment", () => {
    const conversation = getConversation({
      system_template: "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
      system_message: "default",
      roles: { user: "[INST]", assistant: "[/INST]", tool: "[INST]" },
      seps: [" "],
      stop_str: [],
      stop_token_ids: [],
    });
    expect(() => conversation.appendDrowseSystemMessage("later")).toThrow(
      "requires a single-separator system template",
    );
  });
});

test("Gemma turn templates preserve named Drowse roles", () => {
  const conversation = getConversation({
    system_template: "{system_message}",
    system_message: "",
    roles: {
      user: "<start_of_turn>user",
      assistant: "<start_of_turn>model",
      tool: "<start_of_turn>tool",
    },
    role_templates: {
      user: "{user_message}",
      assistant: "{assistant_message}",
    },
    role_content_sep: "\n",
    role_empty_sep: "\n",
    seps: ["<end_of_turn>\n"],
    stop_str: ["<end_of_turn>"],
    stop_token_ids: [1],
  });

  expect(conversation.supportsDrowseNamedRoles()).toBe(true);
  conversation.appendMessage(Role.user, "hello");
  conversation.appendReplyHeader(Role.assistant, "forest_guide");
  expect(conversation.getPromptArray().join("")).toEqual(
    "<start_of_turn>user\nhello<end_of_turn>\n" +
      "<start_of_turn>forest guide\n",
  );
});

test("Llama 3 turn templates preserve named Drowse roles", () => {
  const conversation = getConversation({
    system_template:
      "<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
    system_message: "",
    roles: {
      user: "<|start_header_id|>user",
      assistant: "<|start_header_id|>assistant",
      tool: "<|start_header_id|>ipython",
    },
    role_templates: {
      user: "{user_message}",
      assistant: "{assistant_message}",
    },
    role_content_sep: "<|end_header_id|>\n\n",
    role_empty_sep: "<|end_header_id|>\n\n",
    seps: ["<|eot_id|>"],
    stop_str: [],
    stop_token_ids: [1],
  });

  expect(conversation.supportsDrowseNamedRoles()).toBe(true);
  conversation.appendMessage(Role.user, "hello", "curious_user", true);
  conversation.appendReplyHeader(Role.assistant, "forest_guide");
  expect(conversation.getPromptArray().join("")).toEqual(
    "<|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>" +
      "<|start_header_id|>curious user<|end_header_id|>\n\nhello<|eot_id|>" +
      "<|start_header_id|>forest guide<|end_header_id|>\n\n",
  );
});

test("non-ChatML templates report named roles unsupported", () => {
  const config = JSON.parse(llama2ChatConfigJSONString) as ChatConfig;
  const conversation = getConversation(config.conv_template);
  expect(conversation.supportsDrowseNamedRoles()).toBe(false);
  expect(conversation.supportsDrowseUserSeatGeneration()).toBe(true);
  conversation.appendMessage(Role.user, "hello");
  expect(() =>
    conversation.appendReplyHeader(Role.assistant, "named_assistant"),
  ).toThrow("require a supported role-prefixed conversation template");
});

describe("Test getConversationFromChatCompletionRequest with image", () => {
  // Constants for testing
  type ImageURL = ChatCompletionContentPartImage.ImageURL;
  const dummySystemPromptStr = "dummy system prompt.";
  const dummyRequestStr = "dummy request.";
  const dummyResponseStr = "dummy response.";
  const imageUrl1 = "https://url1";
  const imageUrl2 = "https://url2";
  const imageUrl3 = "https://url3";
  const singleImageInputMessages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: dummySystemPromptStr,
    },
    {
      role: "user",
      content: [
        { type: "text", text: dummyRequestStr },
        {
          type: "image_url",
          image_url: {
            url: imageUrl1,
          },
        },
      ],
    },
  ];

  // system message, single-image user, assistant response, multi-image user
  const multiImageMultiRoundInputMessages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: dummySystemPromptStr,
    },
    {
      role: "user",
      content: [
        { type: "text", text: dummyRequestStr },
        {
          type: "image_url",
          image_url: {
            url: imageUrl1,
          },
        },
      ],
    },
    {
      role: "assistant",
      content: dummyResponseStr,
    },
    {
      role: "user",
      content: [
        { type: "text", text: dummyRequestStr },
        {
          type: "image_url",
          image_url: {
            url: imageUrl2,
          },
        },
        {
          type: "image_url",
          image_url: {
            url: imageUrl3,
          },
        },
      ],
    },
  ];

  test("Test compareConversationObject with different image input", () => {
    const config_json = JSON.parse(phi3_5VisionChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    // deep copy
    const messages1 = JSON.parse(JSON.stringify(singleImageInputMessages));
    const messages2 = JSON.parse(JSON.stringify(singleImageInputMessages));
    const messages3 = JSON.parse(JSON.stringify(singleImageInputMessages));
    const messages4 = JSON.parse(JSON.stringify(singleImageInputMessages));
    messages3[1].content[1].image_url.url = "https://a_different_url";
    messages4[1].content[0].text = "a different text";
    const request1: ChatCompletionRequest = { messages: messages1 };
    const request2: ChatCompletionRequest = { messages: messages2 };
    const request3: ChatCompletionRequest = { messages: messages3 };
    const request4: ChatCompletionRequest = { messages: messages4 };
    const conv1 = getConversationFromChatCompletionRequest(
      request1,
      config,
      true,
    );
    const conv2 = getConversationFromChatCompletionRequest(
      request2,
      config,
      true,
    );
    const conv3 = getConversationFromChatCompletionRequest(
      request3,
      config,
      true,
    );
    const conv4 = getConversationFromChatCompletionRequest(
      request4,
      config,
      true,
    );
    expect(compareConversationObject(conv1, conv2)).toEqual(true);
    expect(compareConversationObject(conv1, conv3)).toEqual(false);
    expect(compareConversationObject(conv2, conv3)).toEqual(false);
    expect(compareConversationObject(conv1, conv4)).toEqual(false);
  });

  test("Test getPromptArray with ContentPart array but only a single text", () => {
    // This should be equivalent to `content: dummyRequestStr`
    const config_json = JSON.parse(phi3_5VisionChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    const messages1: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: dummySystemPromptStr,
      },
      {
        role: "user",
        content: [{ type: "text", text: dummyRequestStr }],
      },
    ];
    const messages2: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: dummySystemPromptStr,
      },
      {
        role: "user",
        content: dummyRequestStr,
      },
    ];
    const request1: ChatCompletionRequest = { messages: messages1 };
    const request2: ChatCompletionRequest = { messages: messages2 };
    const conv1 = getConversationFromChatCompletionRequest(
      request1,
      config,
      true,
    );
    const conv2 = getConversationFromChatCompletionRequest(
      request2,
      config,
      true,
    );
    expect(conv1.getPromptArray()).toEqual([
      dummySystemPromptStr,
      `<|user|>\n${dummyRequestStr}<|end|>\n`,
    ]);
    expect(conv1.getPromptArray()).toEqual(conv2.getPromptArray());
  });

  test("Test getPromptArray with single image input", () => {
    const config_json = JSON.parse(phi3_5VisionChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    const messages1 = JSON.parse(JSON.stringify(singleImageInputMessages));
    const request1: ChatCompletionRequest = { messages: messages1 };
    const conv1 = getConversationFromChatCompletionRequest(
      request1,
      config,
      true,
    );
    expect(conv1.getPromptArray(config)).toEqual([
      dummySystemPromptStr, // phi3_5-vision does not have system template
      [
        `<|user|>\n`,
        { url: imageUrl1 } as ImageURL,
        `\n`,
        `${dummyRequestStr}<|end|>\n`,
      ],
    ]);
  });

  test("Test multiple round with multiple image input, with reply header", () => {
    const config_json = JSON.parse(phi3_5VisionChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    const messages1 = JSON.parse(
      JSON.stringify(multiImageMultiRoundInputMessages),
    );
    const request1: ChatCompletionRequest = { messages: messages1 };
    const conv1 = getConversationFromChatCompletionRequest(
      request1,
      config,
      true,
    );
    conv1.appendReplyHeader(Role.assistant);
    expect(conv1.getPromptArray(config)).toEqual([
      dummySystemPromptStr, // phi3_5-vision does not have system template
      [
        `<|user|>\n`,
        { url: imageUrl1 } as ImageURL,
        `\n`,
        `${dummyRequestStr}<|end|>\n`,
      ],
      `<|assistant|>\n${dummyResponseStr}<|end|>\n`,
      [
        `<|user|>\n`,
        { url: imageUrl2 } as ImageURL,
        `\n`,
        { url: imageUrl3 } as ImageURL,
        `\n`,
        `${dummyRequestStr}<|end|>\n`,
      ],
      `<|assistant|>\n`,
    ]);
    expect(conv1.getPromptArrayLastRound(config)).toEqual([
      [
        `<|user|>\n`,
        { url: imageUrl2 } as ImageURL,
        `\n`,
        { url: imageUrl3 } as ImageURL,
        `\n`,
        `${dummyRequestStr}<|end|>\n`,
      ],
      `<|assistant|>\n`,
    ]);
  });
});
