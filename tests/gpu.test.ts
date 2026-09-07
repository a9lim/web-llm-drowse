/// <reference types="@webgpu/types" />

import { requestGPUDeviceFromAdapter } from "../src/gpu";
import { jest, test, expect } from "@jest/globals";

function adapter(overrides: Record<string, number> = {}) {
  const device = { label: "fixture" };
  const requestDevice = jest.fn<GPUAdapter["requestDevice"]>();
  requestDevice.mockResolvedValue(device as GPUDevice);
  return {
    adapter: {
      features: new Set(["shader-f16", "subgroups"]),
      info: { vendor: "fixture" },
      limits: {
        maxBufferSize: 1 << 30,
        maxStorageBufferBindingSize: 1 << 30,
        maxComputeWorkgroupStorageSize: 32 << 10,
        maxStorageBuffersPerShaderStage: 10,
        maxComputeWorkgroupSizeX: 256,
        maxComputeInvocationsPerWorkgroup: 256,
        ...overrides,
      },
      requestDevice,
    } as unknown as GPUAdapter,
    device,
    requestDevice,
  };
}

test("requests the production device from the supplied adapter", async () => {
  const fixture = adapter();
  const result = await requestGPUDeviceFromAdapter(fixture.adapter);
  expect(result.adapter).toBe(fixture.adapter);
  expect(result.device).toBe(fixture.device);
  expect(result.adapterInfo.vendor).toBe("fixture");
  expect(fixture.requestDevice).toHaveBeenCalledWith({
    requiredLimits: {
      maxBufferSize: 1 << 30,
      maxStorageBufferBindingSize: 1 << 30,
      maxComputeWorkgroupStorageSize: 32 << 10,
      maxStorageBuffersPerShaderStage: 10,
      maxComputeWorkgroupSizeX: 256,
      maxComputeInvocationsPerWorkgroup: 256,
    },
    requiredFeatures: ["shader-f16", "subgroups"],
  });
});

test("uses the documented fallback limits without changing adapters", async () => {
  const fixture = adapter({
    maxBufferSize: 1 << 28,
    maxStorageBufferBindingSize: 1 << 27,
  });
  await requestGPUDeviceFromAdapter(fixture.adapter);
  expect(fixture.requestDevice).toHaveBeenCalledWith(
    expect.objectContaining({
      requiredLimits: expect.objectContaining({
        maxBufferSize: 1 << 28,
        maxStorageBufferBindingSize: 1 << 27,
        maxComputeWorkgroupSizeX: 256,
        maxComputeInvocationsPerWorkgroup: 256,
      }),
    }),
  );
});

test("rejects an adapter below the runtime minimum", async () => {
  const fixture = adapter({ maxStorageBufferBindingSize: (1 << 27) - 1 });
  await expect(requestGPUDeviceFromAdapter(fixture.adapter)).rejects.toThrow(
    "maxStorageBufferBindingSize limit is too small",
  );
  expect(fixture.requestDevice).not.toHaveBeenCalled();
});

test("rejects fewer than ten storage buffers per shader stage", async () => {
  const fixture = adapter({ maxStorageBuffersPerShaderStage: 9 });
  await expect(requestGPUDeviceFromAdapter(fixture.adapter)).rejects.toThrow(
    "maxStorageBuffersPerShaderStage limit is too small",
  );
  expect(fixture.requestDevice).not.toHaveBeenCalled();
});

test("rejects a workgroup X limit below the portable exact readout requirement", async () => {
  const fixture = adapter({ maxComputeWorkgroupSizeX: 255 });
  await expect(requestGPUDeviceFromAdapter(fixture.adapter)).rejects.toThrow(
    "maxComputeWorkgroupSizeX limit is too small: 255 < 256",
  );
  expect(fixture.requestDevice).not.toHaveBeenCalled();
});

test("rejects fewer than 256 compute invocations per workgroup", async () => {
  const fixture = adapter({ maxComputeInvocationsPerWorkgroup: 255 });
  await expect(requestGPUDeviceFromAdapter(fixture.adapter)).rejects.toThrow(
    "maxComputeInvocationsPerWorkgroup limit is too small: 255 < 256",
  );
  expect(fixture.requestDevice).not.toHaveBeenCalled();
});
