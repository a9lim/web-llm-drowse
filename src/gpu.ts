/// <reference types="@webgpu/types" />

import type { GPUDeviceDetectOutput } from "@mlc-ai/web-runtime";

const WEBLLM_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE = 10;
const DROWSE_REQUIRED_MAX_COMPUTE_WORKGROUP_SIZE_X = 256;
const DROWSE_REQUIRED_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP = 256;

export async function requestGPUDeviceFromAdapter(
  adapter: GPUAdapter,
): Promise<GPUDeviceDetectOutput> {
  const maxBufferSize = selectLimit(adapter, "maxBufferSize", 1 << 30, 1 << 28);
  const maxStorageBufferBindingSize = selectLimit(
    adapter,
    "maxStorageBufferBindingSize",
    1 << 30,
    1 << 27,
  );
  const maxComputeWorkgroupStorageSize = requireLimit(
    adapter,
    "maxComputeWorkgroupStorageSize",
    32 << 10,
  );
  const maxStorageBuffersPerShaderStage = requireLimit(
    adapter,
    "maxStorageBuffersPerShaderStage",
    WEBLLM_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
  );
  const maxComputeWorkgroupSizeX = requireLimit(
    adapter,
    "maxComputeWorkgroupSizeX",
    DROWSE_REQUIRED_MAX_COMPUTE_WORKGROUP_SIZE_X,
  );
  const maxComputeInvocationsPerWorkgroup = requireLimit(
    adapter,
    "maxComputeInvocationsPerWorkgroup",
    DROWSE_REQUIRED_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP,
  );
  const requiredFeatures: GPUFeatureName[] = [];
  if (adapter.features.has("shader-f16")) requiredFeatures.push("shader-f16");
  if (adapter.features.has("subgroups")) requiredFeatures.push("subgroups");
  const adapterInfo =
    adapter.info || (await (adapter as any).requestAdapterInfo());
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize,
      maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize,
      maxStorageBuffersPerShaderStage,
      maxComputeWorkgroupSizeX,
      maxComputeInvocationsPerWorkgroup,
    },
    requiredFeatures,
  });
  return { adapter, adapterInfo, device };
}

function selectLimit(
  adapter: GPUAdapter,
  name: "maxBufferSize" | "maxStorageBufferBindingSize",
  preferred: number,
  minimum: number,
): number {
  const supported = adapter.limits[name];
  if (supported >= preferred) return preferred;
  if (supported >= minimum) return minimum;
  throw new Error(
    `The selected WebGPU adapter ${name} limit is too small: ${supported} < ${minimum}`,
  );
}

function requireLimit<
  T extends
    | "maxComputeWorkgroupStorageSize"
    | "maxStorageBuffersPerShaderStage"
    | "maxComputeWorkgroupSizeX"
    | "maxComputeInvocationsPerWorkgroup",
>(adapter: GPUAdapter, name: T, required: number): number {
  const supported = adapter.limits[name];
  if (supported < required) {
    throw new Error(
      `The selected WebGPU adapter ${name} limit is too small: ${supported} < ${required}`,
    );
  }
  return required;
}
