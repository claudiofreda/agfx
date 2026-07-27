/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-27 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "device info driver version not empty".
//
// agfxDeviceInfo::driverVersion is what a caller diffs against a version stored alongside a
// serialized pipeline cache blob to decide whether the driver has changed since the blob was built
// (agfxRenderPipelineCreate/agfxComputePipelineCreate silently ignore a stale one rather than
// failing, so there is no other signal). An empty or unpopulated string would make that comparison
// meaningless, so this just pins down that a real (non-WARP) device reports one.

#include "agfx_tests/test_gpu.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>

AGFX_TEST_VALIDATION(DeviceInfoDriverVersionNotEmpty, C)
{
    using namespace agfxtest;

    const agfxDeviceCreateInfo deviceInfo = DefaultDeviceCreateInfo();
    agfxDevice* device = agfxDeviceCreate(&deviceInfo);
    AGFX_EXPECT_NOT_NULL(device);

    agfxDeviceInfo info{};
    agfxDeviceGetInfo(device, &info);
    agfxDeviceDestroy(device);

    AGFX_EXPECT_MSG(strlen(info.driverVersion) > 0, "driverVersion was empty");
    AGFX_EXPECT_MSG(strcmp(info.driverVersion, "Unknown") != 0, "driverVersion fell back to \"Unknown\"");
}

AGFX_TEST_VALIDATION(DeviceInfoDriverVersionNotEmpty, Cpp)
{
    using namespace agfxtest;

    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    const agfxDeviceInfo info = device.GetInfo();
    AGFX_EXPECT_MSG(strlen(info.driverVersion) > 0, "driverVersion was empty");
    AGFX_EXPECT_MSG(strcmp(info.driverVersion, "Unknown") != 0, "driverVersion fell back to \"Unknown\"");
}

AGFX_TEST_VALIDATION(DeviceInfoDriverVersionNotEmpty, Ez)
{
    using namespace agfxtest;

    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless
    contextInfo.width = 128;
    contextInfo.height = 128;
    agfx::ez::Context context(contextInfo);

    const agfxDeviceInfo info = context.GetDevice().GetInfo();
    AGFX_EXPECT_MSG(strlen(info.driverVersion) > 0, "driverVersion was empty");
    AGFX_EXPECT_MSG(strcmp(info.driverVersion, "Unknown") != 0, "driverVersion fell back to \"Unknown\"");
}
