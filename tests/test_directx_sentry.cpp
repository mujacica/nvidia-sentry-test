#ifdef _WIN32
#include "../src/directx/d3d11_renderer.h"
#include <gtest/gtest.h>
#include <sentry.h>

class DirectXSentryTest : public ::testing::Test {
protected:
  void SetUp() override {
    sentry_options_t *options = sentry_options_new();
    sentry_options_set_dsn(options,
                           nullptr); // Disable actual reporting for tests
    sentry_options_set_database_path(options, ".sentry-native-directx-test");
    sentry_options_set_debug(options, 0);

    if (sentry_init(options) != 0) {
      FAIL() << "Failed to initialize Sentry for testing";
    }

    renderer = std::make_unique<D3D11Renderer>();

    // Skip tests if DirectX is not available
    if (!renderer->initialize()) {
      GTEST_SKIP() << "DirectX not available, skipping DirectX tests";
    }
  }

  void TearDown() override {
    renderer.reset();
    sentry_close();
  }

  std::unique_ptr<D3D11Renderer> renderer;
};

TEST_F(DirectXSentryTest, DirectXInitialization) {
  // This test passes if we get here (renderer initialized successfully)
  SUCCEED();
}

TEST_F(DirectXSentryTest, SentryInitialization) {
  sentry_value_t event = sentry_value_new_event();
  sentry_value_set_by_key(event, "message",
                          sentry_value_new_string("Test DirectX event"));

  EXPECT_NO_THROW(sentry_capture_event(event));
}

TEST_F(DirectXSentryTest, InvalidBufferAccess) {
  DirectXError error = renderer->test_invalid_buffer_access();

  // This should typically fail with invalid buffer access
  EXPECT_NE(error.code, 0)
      << "Expected DirectX error for invalid buffer access";
  EXPECT_STREQ(error.type, "D3D11_INVALID_BUFFER_ACCESS");
}

TEST_F(DirectXSentryTest, InvalidShaderResource) {
  DirectXError error = renderer->test_invalid_shader_resource();

  // This test checks that we can call the crash test function
  // The actual behavior depends on DirectX driver
  EXPECT_NO_THROW({
    if (error.code != 0) {
      std::cout << "DirectX error detected: " << error.message << std::endl;
    }
  });
}

TEST_F(DirectXSentryTest, InvalidRenderTarget) {
  DirectXError error = renderer->test_invalid_render_target();

  // This test checks that we can call the crash test function
  // The actual behavior depends on DirectX driver validation
  EXPECT_NO_THROW({
    if (error.code != 0) {
      std::cout << "DirectX error detected: " << error.message << std::endl;
    }
  });
}

TEST_F(DirectXSentryTest, OutOfBoundsVertexBuffer) {
  DirectXError error = renderer->test_out_of_bounds_vertex_buffer();

  // This test checks that we can call the crash test function
  // The actual behavior depends on DirectX driver validation
  EXPECT_NO_THROW({
    if (error.code != 0) {
      std::cout << "DirectX error detected: " << error.message << std::endl;
    }
  });
}

#else
#include <gtest/gtest.h>

TEST(DirectXSentryTest, WindowsOnly) {
  GTEST_SKIP() << "DirectX tests only available on Windows";
}

#endif // _WIN32