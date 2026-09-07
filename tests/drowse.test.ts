import { describe, expect, test } from "@jest/globals";
import {
  DROWSE_HOOK_ABI,
  DROWSE_EXACT_READOUT_ABI,
  DROWSE_EXACT_READOUT_ABI_VERSION,
  DROWSE_EXACT_READOUT_MAX_SAE_FEATURES_PER_CHUNK,
  DROWSE_READOUT_TOP_K,
  DROWSE_STRUCTURED_HOOK_FORMAT,
  DROWSE_STRUCTURED_HOOK_FORMAT_V2,
  DROWSE_STRUCTURED_MAX_AFFINE_GROUPS,
  DROWSE_STRUCTURED_MAX_CURVE_NODES,
  DROWSE_STRUCTURED_MAX_CURVES,
  DROWSE_STRUCTURED_MAX_EMBED_DIM,
  DROWSE_STRUCTURED_MAX_INTRINSIC_DIM,
  DROWSE_STRUCTURED_MAX_PROBES,
  DROWSE_STRUCTURED_MAX_RANK,
  DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE,
  DROWSE_STRUCTURED_PROFILE_MAGIC,
  DROWSE_STRUCTURED_MIN_COMPUTE_WORKGROUP_STORAGE_SIZE,
  DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES,
  DROWSE_STRUCTURED_MAX_WHITENER_RANK,
  DROWSE_STRUCTURED_MAX_GEOMETRY_CANDIDATES,
  DROWSE_STRUCTURED_GEOMETRY_OUTPUT_STRIDE,
  DROWSE_STRUCTURED_GEOMETRY_FOOT_RESTARTS,
  DROWSE_STRUCTURED_GEOMETRY_FOOT_ITERATIONS,
  DROWSE_STRUCTURED_GEOMETRY_WARM_RESTARTS,
  DROWSE_STRUCTURED_GEOMETRY_WARM_ITERATIONS,
  DROWSE_STRUCTURED_GEOMETRY_HEADER_STRIDE,
  DROWSE_STRUCTURED_GEOMETRY_KERNEL_STORAGE_BINDINGS,
  DROWSE_STRUCTURED_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
  DrowseRankOneProgram,
  DrowseStructuredProgram,
  packDrowseCurveParameters,
  packDrowseGeometryHeader,
  packDrowseGeometryPayload,
  drowseGeometryPayloadElements,
  updatePackedDrowseCurveActive,
  validateDrowseCapturePositions,
  validateDrowseCaptureRows,
  validateDrowseRankOneProgram,
  validateDrowseJlensDictionary,
  validateDrowseSaeDictionary,
  validateDrowseStructuredProgram,
  decodeDrowseStructuredHookProfile,
} from "../src/drowse";

describe("Drowse structured hook profile", () => {
  const descriptor = Int32Array.from([
    DROWSE_STRUCTURED_PROFILE_MAGIC,
    3,
    3,
    4,
    3,
    12,
    768,
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
    drowseGeometryPayloadElements(12, 768),
    DROWSE_EXACT_READOUT_ABI_VERSION,
    DROWSE_READOUT_TOP_K,
    DROWSE_EXACT_READOUT_MAX_SAE_FEATURES_PER_CHUNK,
  ]);

  test("decodes the immutable compiled descriptor", () => {
    expect(decodeDrowseStructuredHookProfile(descriptor, 12, 768)).toEqual({
      schemaVersion: 3,
      id: "standard-v3",
      hookAbi: DROWSE_HOOK_ABI,
      format: DROWSE_STRUCTURED_HOOK_FORMAT,
      layerCount: 12,
      hiddenSize: 768,
      maxAffineGroups: DROWSE_STRUCTURED_MAX_AFFINE_GROUPS,
      maxRank: DROWSE_STRUCTURED_MAX_RANK,
      maxProbes: DROWSE_STRUCTURED_MAX_PROBES,
      maxCurves: DROWSE_STRUCTURED_MAX_CURVES,
      maxCurveNodes: DROWSE_STRUCTURED_MAX_CURVE_NODES,
      maxIntrinsicDim: DROWSE_STRUCTURED_MAX_INTRINSIC_DIM,
      maxEmbedDim: DROWSE_STRUCTURED_MAX_EMBED_DIM,
      curveParameterStride: DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE,
      maxComputeWorkgroupStorageSize:
        DROWSE_STRUCTURED_MIN_COMPUTE_WORKGROUP_STORAGE_SIZE,
      maxGeometryProbes: DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES,
      maxWhitenerRank: DROWSE_STRUCTURED_MAX_WHITENER_RANK,
      maxGeometryCandidates: DROWSE_STRUCTURED_MAX_GEOMETRY_CANDIDATES,
      geometryOutputStride: DROWSE_STRUCTURED_GEOMETRY_OUTPUT_STRIDE,
      geometryFootRestarts: DROWSE_STRUCTURED_GEOMETRY_FOOT_RESTARTS,
      geometryFootIterations: DROWSE_STRUCTURED_GEOMETRY_FOOT_ITERATIONS,
      geometryWarmRestarts: DROWSE_STRUCTURED_GEOMETRY_WARM_RESTARTS,
      geometryWarmIterations: DROWSE_STRUCTURED_GEOMETRY_WARM_ITERATIONS,
      requiredMaxStorageBuffersPerShaderStage:
        DROWSE_STRUCTURED_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
      geometryKernelStorageBindings:
        DROWSE_STRUCTURED_GEOMETRY_KERNEL_STORAGE_BINDINGS,
      geometryHeaderStride: DROWSE_STRUCTURED_GEOMETRY_HEADER_STRIDE,
      geometryPayloadElements: drowseGeometryPayloadElements(12, 768),
      exactReadoutAbiVersion: DROWSE_EXACT_READOUT_ABI_VERSION,
      exactReadoutAbi: DROWSE_EXACT_READOUT_ABI,
      readoutTopK: DROWSE_READOUT_TOP_K,
      maxSaeFeaturesPerChunk: DROWSE_EXACT_READOUT_MAX_SAE_FEATURES_PER_CHUNK,
    });
  });

  test("rejects capacity or model-shape drift", () => {
    const changed = descriptor.slice();
    changed[8] += 1;
    expect(() => decodeDrowseStructuredHookProfile(changed, 12, 768)).toThrow();
    const oldProfile = descriptor.slice();
    oldProfile[2] = 2;
    expect(() =>
      decodeDrowseStructuredHookProfile(oldProfile, 12, 768),
    ).toThrow();
    expect(() =>
      decodeDrowseStructuredHookProfile(descriptor, 11, 768),
    ).toThrow();
  });
});

describe("Drowse precomputed SAE dictionaries", () => {
  const dictionary = {
    hookAbi: DROWSE_HOOK_ABI,
    bindingId: "a".repeat(64),
    hiddenSize: 2,
    runtimeLayerIndex: 1,
    featureCount: 3,
    activation: "relu",
    encoder: Float32Array.from([1, 2, 3, 4, 5, 6]),
    encoderBias: Float32Array.from([0.1, 0.2, 0.3]),
    encoderThreshold: new Float32Array(3),
    decoderBias: Float32Array.from([0.4, 0.5]),
  } as const;

  test("validates the exact runtime shape and finite payload", () => {
    expect(() => validateDrowseSaeDictionary(dictionary, 2, 2)).not.toThrow();
    expect(() =>
      validateDrowseSaeDictionary(
        { ...dictionary, runtimeLayerIndex: 2 },
        2,
        2,
      ),
    ).toThrow(/layer is invalid/);
    expect(() =>
      validateDrowseSaeDictionary({ ...dictionary, hiddenSize: 3 }, 2, 2),
    ).toThrow(/targets 3 hidden dimensions/);
    expect(() =>
      validateDrowseSaeDictionary(
        {
          ...dictionary,
          encoderBias: Float32Array.from([0.1, Number.NaN, 0.3]),
        },
        2,
        2,
      ),
    ).toThrow(/encoder bias values must be finite/);
    expect(() =>
      validateDrowseSaeDictionary(
        { ...dictionary, encoder: new Float32Array(5) },
        2,
        2,
      ),
    ).toThrow(/encoder must contain exactly 6 values/);
  });
});

describe("Drowse resident J-lens dictionaries", () => {
  const dictionary = {
    hookAbi: DROWSE_HOOK_ABI,
    bindingId: "b".repeat(64),
    hiddenSize: 2,
    layerIndices: Int32Array.from([0, 1]),
    matrices: [new Float32Array(4), new Float32Array(4)],
  } as const;

  test("validates exact bindings, layers, and per-layer matrix shapes", () => {
    expect(() => validateDrowseJlensDictionary(dictionary, 2, 2)).not.toThrow();
    expect(() =>
      validateDrowseJlensDictionary(
        { ...dictionary, bindingId: "floating" },
        2,
        2,
      ),
    ).toThrow(/identity is invalid/);
    expect(() =>
      validateDrowseJlensDictionary(
        { ...dictionary, layerIndices: Int32Array.from([1, 0]) },
        2,
        2,
      ),
    ).toThrow(/unique, ordered, and in range/);
    expect(() =>
      validateDrowseJlensDictionary(
        { ...dictionary, matrices: [new Float32Array(4), new Float32Array(3)] },
        2,
        2,
      ),
    ).toThrow(/wrong shape/);
  });
});

function program(): DrowseRankOneProgram {
  return {
    hookAbi: DROWSE_HOOK_ABI,
    hiddenSize: 2,
    layerCount: 2,
    enabled: new Uint32Array([1, 0]),
    basis: new Float32Array([1, 0, 0, 1]),
    neutral: new Float32Array(4),
    target: new Float32Array([0.5, -0.5]),
    along: new Float32Array([1, 1]),
    collapse: new Float32Array([1, 1]),
    probeBasis: new Float32Array([1, 0, 0, 1]),
    probeNeutral: new Float32Array(4),
  };
}

function structuredProgram(): DrowseStructuredProgram {
  const layerCount = 2;
  const hiddenSize = 2;
  const curveSlots = layerCount * DROWSE_STRUCTURED_MAX_CURVES;
  return {
    hookAbi: DROWSE_HOOK_ABI,
    format: DROWSE_STRUCTURED_HOOK_FORMAT_V2,
    hiddenSize,
    layerCount,
    affineActive: new Uint32Array(
      layerCount * DROWSE_STRUCTURED_MAX_AFFINE_GROUPS,
    ),
    affineBasis: new Float32Array(
      layerCount *
        DROWSE_STRUCTURED_MAX_AFFINE_GROUPS *
        DROWSE_STRUCTURED_MAX_RANK *
        hiddenSize,
    ),
    affineNeutral: new Float32Array(
      layerCount * DROWSE_STRUCTURED_MAX_AFFINE_GROUPS * hiddenSize,
    ),
    affineTarget: new Float32Array(
      layerCount *
        DROWSE_STRUCTURED_MAX_AFFINE_GROUPS *
        DROWSE_STRUCTURED_MAX_RANK,
    ),
    affineAlong: new Float32Array(
      layerCount * DROWSE_STRUCTURED_MAX_AFFINE_GROUPS,
    ),
    affineKappa: new Float32Array(
      layerCount *
        DROWSE_STRUCTURED_MAX_AFFINE_GROUPS *
        DROWSE_STRUCTURED_MAX_RANK,
    ),
    probeKind: new Uint32Array(layerCount * DROWSE_STRUCTURED_MAX_PROBES),
    probeDirection: new Float32Array(
      layerCount * DROWSE_STRUCTURED_MAX_PROBES * hiddenSize,
    ),
    probeBias: new Float32Array(layerCount * DROWSE_STRUCTURED_MAX_PROBES),
    probeThreshold: new Float32Array(layerCount * DROWSE_STRUCTURED_MAX_PROBES),
    curveActive: new Uint32Array(curveSlots),
    curveRank: new Uint32Array(curveSlots),
    curveIntrinsicDim: new Uint32Array(curveSlots),
    curveEmbedDim: new Uint32Array(curveSlots),
    curveNodeCount: new Uint32Array(curveSlots),
    curveBasis: new Float32Array(
      curveSlots * DROWSE_STRUCTURED_MAX_RANK * hiddenSize,
    ),
    curveNeutral: new Float32Array(curveSlots * hiddenSize),
    curveNodeParameters: new Float32Array(
      curveSlots *
        DROWSE_STRUCTURED_MAX_CURVE_NODES *
        DROWSE_STRUCTURED_MAX_EMBED_DIM,
    ),
    curveRbfWeights: new Float32Array(
      curveSlots *
        DROWSE_STRUCTURED_MAX_CURVE_NODES *
        DROWSE_STRUCTURED_MAX_RANK,
    ),
    curvePolynomial: new Float32Array(
      curveSlots *
        (DROWSE_STRUCTURED_MAX_EMBED_DIM + 1) *
        DROWSE_STRUCTURED_MAX_RANK,
    ),
    curveCoordinateOffset: new Float32Array(
      curveSlots * DROWSE_STRUCTURED_MAX_EMBED_DIM,
    ),
    curveCoordinateScale: new Float32Array(
      curveSlots * DROWSE_STRUCTURED_MAX_EMBED_DIM,
    ),
    curveOrigin: new Float32Array(
      curveSlots * DROWSE_STRUCTURED_MAX_INTRINSIC_DIM,
    ),
    curveTarget: new Float32Array(
      curveSlots * DROWSE_STRUCTURED_MAX_INTRINSIC_DIM,
    ),
    curveAlong: new Float32Array(curveSlots),
    curveOnto: new Float32Array(curveSlots),
    curveBounds: new Float32Array(
      curveSlots * DROWSE_STRUCTURED_MAX_INTRINSIC_DIM * 2,
    ),
    curveAxisPeriodic: new Uint32Array(
      curveSlots * DROWSE_STRUCTURED_MAX_INTRINSIC_DIM,
    ),
    curveAxisPeriod: new Float32Array(
      curveSlots * DROWSE_STRUCTURED_MAX_INTRINSIC_DIM,
    ),
    curveSigmaPresent: new Uint32Array(curveSlots),
    curveSigmaRbfWeights: new Float32Array(
      curveSlots * DROWSE_STRUCTURED_MAX_CURVE_NODES,
    ),
    curveSigmaPolynomial: new Float32Array(
      curveSlots * (DROWSE_STRUCTURED_MAX_EMBED_DIM + 1),
    ),
    curveDamping: new Float32Array(curveSlots),
  };
}

function structuredGeometryProgram(): DrowseStructuredProgram {
  const program = structuredProgram();
  const l = program.layerCount;
  const d = program.hiddenSize;
  const g = DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES;
  const r = DROWSE_STRUCTURED_MAX_RANK;
  const i = DROWSE_STRUCTURED_MAX_INTRINSIC_DIM;
  const n = DROWSE_STRUCTURED_MAX_CURVE_NODES;
  const slots = l * g;
  program.format = DROWSE_STRUCTURED_HOOK_FORMAT;
  program.curveDomainKind = new Uint32Array(l * DROWSE_STRUCTURED_MAX_CURVES);
  program.whitenerRank = new Uint32Array(l);
  program.whitenerRidge = new Float32Array(l).fill(1);
  program.whitenerBasis = new Float32Array(
    l * DROWSE_STRUCTURED_MAX_WHITENER_RANK * d,
  );
  program.whitenerCorrection = new Float32Array(
    l * DROWSE_STRUCTURED_MAX_WHITENER_RANK,
  );
  program.geometryActive = new Uint32Array(slots);
  program.geometryKind = new Uint32Array(slots);
  program.geometryRank = new Uint32Array(slots);
  program.geometryIntrinsicDim = new Uint32Array(slots);
  program.geometryCandidateCount = new Uint32Array(slots);
  program.geometryCurveNodeCount = new Uint32Array(slots);
  program.geometryDomainKind = new Uint32Array(slots);
  program.geometryMean = new Float32Array(slots * d);
  program.geometryInverseMean = new Float32Array(slots * d);
  program.geometryBasis = new Float32Array(slots * r * d);
  program.geometryGramInverse = new Float32Array(slots * r * r);
  program.geometryCholesky = new Float32Array(slots * r * r);
  program.geometryNodeWhite = new Float32Array(
    slots * DROWSE_STRUCTURED_MAX_GEOMETRY_CANDIDATES * r,
  );
  program.geometryCoordMap = new Float32Array(slots * i * r);
  program.geometryCoordBias = new Float32Array(slots * i);
  program.geometryCurveParameters = new Float32Array(
    slots * DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE,
  );
  program.geometryCurveNodeCoords = new Float32Array(slots * n * i);
  program.geometryCurveNodeValues = new Float32Array(slots * n * r);
  program.geometryFeet = new Float32Array(slots * i);
  return program;
}

describe("Drowse rank-one hook programs", () => {
  test("accepts exact finite typed buffers", () => {
    expect(() => validateDrowseRankOneProgram(program(), 2, 2)).not.toThrow();
  });

  test("rejects model identity and buffer shape mismatches", () => {
    expect(() => validateDrowseRankOneProgram(program(), 3, 2)).toThrow(
      /model requires 2x3/,
    );
    const invalid = program();
    invalid.basis = new Float32Array(3);
    expect(() => validateDrowseRankOneProgram(invalid, 2, 2)).toThrow(
      /basis must contain exactly 4 values/,
    );
  });

  test("rejects non-binary enable flags and non-finite values", () => {
    const invalidFlag = program();
    invalidFlag.enabled[0] = 2;
    expect(() => validateDrowseRankOneProgram(invalidFlag, 2, 2)).toThrow(
      /enabled values must be zero or one/,
    );
    const invalidFloat = program();
    invalidFloat.target[0] = Number.NaN;
    expect(() => validateDrowseRankOneProgram(invalidFloat, 2, 2)).toThrow(
      /target values must be finite/,
    );
  });

  test("requires normalized active steering and nonzero probe bases", () => {
    const invalidBasis = program();
    invalidBasis.basis[0] = 0.5;
    expect(() => validateDrowseRankOneProgram(invalidBasis, 2, 2)).toThrow(
      /basis layer 0 must be unit length/,
    );
    const invalidProbe = program();
    invalidProbe.probeBasis[2] = 0.5;
    expect(() => validateDrowseRankOneProgram(invalidProbe, 2, 2)).toThrow(
      /probeBasis layer 1 must be unit length/,
    );
    const disabledProbe = program();
    disabledProbe.probeBasis.fill(0);
    expect(() =>
      validateDrowseRankOneProgram(disabledProbe, 2, 2),
    ).not.toThrow();
  });
});

describe("Drowse structured J-lens programs", () => {
  test("requires exact full-vocabulary readout buffers for probability probes", () => {
    const valid = structuredProgram();
    valid.probeKind[8] = 3;
    valid.jLensBindingId = "b".repeat(64);
    valid.jLensLayerIndices = new Int32Array([1]);
    valid.jLensTokenIds = new Int32Array(DROWSE_STRUCTURED_MAX_PROBES);
    expect(() => validateDrowseStructuredProgram(valid, 2, 2)).not.toThrow();

    const missing = structuredProgram();
    missing.probeKind[0] = 3;
    expect(() => validateDrowseStructuredProgram(missing, 2, 2)).toThrow(
      /complete probability program/,
    );

    const fullVocabularyReadout = structuredProgram();
    fullVocabularyReadout.jLensBindingId = "b".repeat(64);
    fullVocabularyReadout.jLensLayerIndices = new Int32Array([0, 1]);
    fullVocabularyReadout.jLensTokenIds = new Int32Array(
      DROWSE_STRUCTURED_MAX_PROBES,
    );
    expect(() =>
      validateDrowseStructuredProgram(fullVocabularyReadout, 2, 2),
    ).not.toThrow();
  });

  test("rejects malformed exact readout bindings and negative token IDs", () => {
    const invalidBinding = structuredProgram();
    invalidBinding.probeKind[8] = 3;
    invalidBinding.jLensBindingId = "floating";
    invalidBinding.jLensLayerIndices = new Int32Array([1]);
    invalidBinding.jLensTokenIds = new Int32Array(DROWSE_STRUCTURED_MAX_PROBES);
    expect(() => validateDrowseStructuredProgram(invalidBinding, 2, 2)).toThrow(
      /binding is invalid/,
    );

    const negative = structuredProgram();
    negative.probeKind[8] = 3;
    negative.jLensBindingId = "b".repeat(64);
    negative.jLensLayerIndices = new Int32Array([1]);
    negative.jLensTokenIds = new Int32Array(DROWSE_STRUCTURED_MAX_PROBES);
    negative.jLensTokenIds[0] = -1;
    expect(() => validateDrowseStructuredProgram(negative, 2, 2)).toThrow(
      /token IDs are invalid/,
    );

    const wrongLayers = structuredProgram();
    wrongLayers.probeKind[8] = 3;
    wrongLayers.jLensBindingId = "b".repeat(64);
    wrongLayers.jLensLayerIndices = new Int32Array([0]);
    wrongLayers.jLensTokenIds = new Int32Array(DROWSE_STRUCTURED_MAX_PROBES);
    expect(() => validateDrowseStructuredProgram(wrongLayers, 2, 2)).toThrow(
      /exactly match the active probability layers/,
    );

    const duplicateLayers = structuredProgram();
    duplicateLayers.probeKind[0] = 3;
    duplicateLayers.probeKind[8] = 3;
    duplicateLayers.jLensBindingId = "b".repeat(64);
    duplicateLayers.jLensLayerIndices = new Int32Array([0, 0]);
    duplicateLayers.jLensTokenIds = new Int32Array(
      DROWSE_STRUCTURED_MAX_PROBES,
    );
    expect(() =>
      validateDrowseStructuredProgram(duplicateLayers, 2, 2),
    ).toThrow(/unique, ordered, and in range/);

    const outOfRangeLayer = structuredProgram();
    outOfRangeLayer.probeKind[8] = 3;
    outOfRangeLayer.jLensBindingId = "b".repeat(64);
    outOfRangeLayer.jLensLayerIndices = new Int32Array([2]);
    outOfRangeLayer.jLensTokenIds = new Int32Array(
      DROWSE_STRUCTURED_MAX_PROBES,
    );
    expect(() =>
      validateDrowseStructuredProgram(outOfRangeLayer, 2, 2),
    ).toThrow(/unique, ordered, and in range/);
  });
});

describe("Drowse structured-v3 geometry programs", () => {
  test("accepts zeroed whiteners when no geometry slot is active", () => {
    const inactive = structuredGeometryProgram();
    inactive.whitenerRidge!.fill(0);
    expect(() => validateDrowseStructuredProgram(inactive, 2, 2)).not.toThrow();
  });

  test("accepts exact flat and hyperspherical curved geometry descriptors", () => {
    const flat = structuredGeometryProgram();
    flat.whitenerRank![0] = 1;
    flat.geometryActive![0] = 1;
    flat.geometryKind![0] = 1;
    flat.geometryRank![0] = 1;
    flat.geometryIntrinsicDim![0] = 1;
    flat.geometryCandidateCount![0] = 1;
    expect(() => validateDrowseStructuredProgram(flat, 2, 2)).not.toThrow();

    const sphere = structuredGeometryProgram();
    sphere.whitenerRank![0] = 1;
    sphere.geometryActive![0] = 1;
    sphere.geometryKind![0] = 2;
    sphere.geometryRank![0] = 3;
    sphere.geometryIntrinsicDim![0] = 2;
    sphere.geometryCandidateCount![0] = 3;
    sphere.geometryCurveNodeCount![0] = 2;
    sphere.geometryDomainKind![0] = 2;
    expect(() => validateDrowseStructuredProgram(sphere, 2, 2)).not.toThrow();
  });

  test("requires a positive-definite whitener only on active geometry layers", () => {
    const missing = structuredGeometryProgram();
    missing.whitenerRidge!.fill(0);
    missing.geometryActive![DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES] = 1;
    missing.geometryKind![DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES] = 1;
    missing.geometryRank![DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES] = 1;
    missing.geometryIntrinsicDim![DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES] = 1;
    missing.geometryCandidateCount![DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES] = 1;
    missing.whitenerRank![1] = 1;
    expect(() => validateDrowseStructuredProgram(missing, 2, 2)).toThrow(
      /active geometry layer 1 requires a positive-definite whitener/,
    );

    missing.whitenerRidge![1] = Number.NaN;
    expect(() => validateDrowseStructuredProgram(missing, 2, 2)).toThrow(
      /whitenerRidge values must be finite/,
    );

    missing.whitenerRidge![1] = 1;
    expect(() => validateDrowseStructuredProgram(missing, 2, 2)).not.toThrow();
  });

  test("rejects missing domain buffers and invalid geometry kinds", () => {
    const missing = structuredGeometryProgram();
    missing.curveDomainKind = undefined;
    expect(() => validateDrowseStructuredProgram(missing, 2, 2)).toThrow(
      /missing curveDomainKind/,
    );

    const invalid = structuredGeometryProgram();
    invalid.whitenerRank![0] = 1;
    invalid.geometryActive![0] = 1;
    invalid.geometryKind![0] = 3;
    invalid.geometryRank![0] = 1;
    invalid.geometryIntrinsicDim![0] = 1;
    invalid.geometryCandidateCount![0] = 1;
    expect(() => validateDrowseStructuredProgram(invalid, 2, 2)).toThrow(
      /geometry slot 0 exceeds the supported shape/,
    );
  });
});

describe("Drowse curved program packing", () => {
  test("preserves the fixed GPU ABI layout and updates only active signals", () => {
    const program = structuredProgram();
    program.curveActive[0] = 1;
    program.curveIntrinsicDim[0] = 2;
    program.curveNodeParameters[0] = 3;
    program.curveRbfWeights[0] = 4;
    program.curvePolynomial[0] = 5;
    program.curveCoordinateOffset[0] = 6;
    program.curveCoordinateScale[0] = 7;
    program.curveOrigin[0] = 8;
    program.curveTarget[0] = 9;
    program.curveAlong[0] = 10;
    program.curveOnto[0] = 11;
    program.curveBounds[0] = 12;
    program.curveAxisPeriodic[0] = 1;
    program.curveAxisPeriod[0] = 14;
    program.curveSigmaPresent[0] = 1;
    program.curveSigmaRbfWeights[0] = 16;
    program.curveSigmaPolynomial[0] = 17;
    program.curveDamping[0] = 18;

    const packed = packDrowseCurveParameters(program);
    expect(packed.length).toBe(
      program.layerCount *
        DROWSE_STRUCTURED_MAX_CURVES *
        DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE,
    );
    for (const [offset, value] of [
      [0, 1],
      [1, 2],
      [2, 3],
      [258, 4],
      [514, 5],
      [586, 6],
      [594, 7],
      [602, 8],
      [606, 9],
      [610, 10],
      [611, 11],
      [612, 12],
      [620, 1],
      [624, 14],
      [628, 1],
      [629, 16],
      [661, 17],
      [670, 18],
    ]) {
      expect(packed[offset]).toBe(value);
    }
    const before = packed.slice();
    const active = new Uint32Array(program.curveActive.length);
    active[1] = 1;
    updatePackedDrowseCurveActive(packed, active);
    expect(packed[0]).toBe(0);
    expect(packed[DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE]).toBe(1);
    expect(packed.slice(1, DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE)).toEqual(
      before.slice(1, DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE),
    );
  });
});

describe("Drowse geometry program packing", () => {
  test("packs one immutable header and payload within the eight-binding budget", () => {
    const program = structuredGeometryProgram();
    const headerFields = [
      program.geometryActive!,
      program.geometryKind!,
      program.geometryRank!,
      program.geometryIntrinsicDim!,
      program.geometryCandidateCount!,
      program.geometryCurveNodeCount!,
      program.geometryDomainKind!,
    ];
    headerFields.forEach((field, index) => {
      field[0] = index + 1;
    });
    const header = packDrowseGeometryHeader(program);
    expect(Array.from(header.slice(0, 7))).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(header.length).toBe(
      program.layerCount *
        DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES *
        DROWSE_STRUCTURED_GEOMETRY_HEADER_STRIDE,
    );

    const payloadFields = [
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
    payloadFields.forEach((field, index) => field.fill(index + 1));
    const payload = packDrowseGeometryPayload(program);
    let offset = 0;
    payloadFields.forEach((field, index) => {
      expect(payload[offset]).toBe(index + 1);
      expect(payload[offset + field.length - 1]).toBe(index + 1);
      offset += field.length;
    });
    expect(offset).toBe(payload.length);
    expect(payload.length).toBe(
      drowseGeometryPayloadElements(program.layerCount, program.hiddenSize),
    );
    expect(DROWSE_STRUCTURED_GEOMETRY_KERNEL_STORAGE_BINDINGS).toBe(8);
    expect(
      DROWSE_STRUCTURED_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
    ).toBe(8);
  });
});

describe("Drowse residual capture positions", () => {
  test("accepts ordered unique positions inside the prompt", () => {
    expect(() => validateDrowseCapturePositions([0, 3, 4], 5)).not.toThrow();
  });

  test.each([
    [[], 5],
    [[0], 0],
    [[-1], 5],
    [[1, 1], 5],
    [[2, 1], 5],
    [[5], 5],
    [[1.5], 5],
  ])("rejects invalid positions %p for %p tokens", (positions, tokenCount) => {
    expect(() =>
      validateDrowseCapturePositions(
        positions as number[],
        tokenCount as number,
      ),
    ).toThrow();
  });
});

describe("Drowse capture row validation", () => {
  test("accepts a conversation ending in an assistant response", () => {
    expect(() =>
      validateDrowseCaptureRows(
        [
          {
            system: "brief",
            messages: [
              { role: "user", content: "prompt" },
              { role: "system", content: "keep this position" },
              { role: "assistant", content: "response" },
            ],
          },
        ],
        [1, 3, 8],
      ),
    ).not.toThrow();
  });

  test("rejects a named system capture turn", () => {
    expect(() =>
      validateDrowseCaptureRows(
        [
          {
            system: "brief",
            messages: [
              { role: "system", content: "later", roleName: "named" },
              { role: "assistant", content: "response" },
            ],
          },
        ],
        [],
      ),
    ).toThrow(/cannot name a role/);
  });

  test("rejects unfinished rows and unsorted special-token IDs", () => {
    expect(() =>
      validateDrowseCaptureRows(
        [{ system: "", messages: [{ role: "user", content: "prompt" }] }],
        [],
      ),
    ).toThrow(/end with an assistant/);
    expect(() =>
      validateDrowseCaptureRows(
        [
          {
            system: "",
            messages: [{ role: "assistant", content: "response" }],
          },
        ],
        [3, 3],
      ),
    ).toThrow(/sorted unique/);
  });
});
