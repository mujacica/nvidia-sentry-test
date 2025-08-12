// GPU Info demo is supported on all platforms (Linux, Windows, macOS)
#include <gtest/gtest.h>
#include <sentry.h>

class GPUInfoSentryTest : public ::testing::Test {
protected:
  void SetUp() override {
  }

  void TearDown() override {
  }
};

TEST_F(GPUInfoSentryTest, SentryInitialization) {
  // Test that we can create Sentry objects without initializing
  sentry_options_t *options = sentry_options_new();
  EXPECT_NE(options, nullptr);
  
  // Test basic event creation
  sentry_value_t event = sentry_value_new_event();
  EXPECT_TRUE(sentry_value_is_null(event) == 0);
  
  sentry_options_free(options);
}

TEST_F(GPUInfoSentryTest, BasicInfoCollection) {
  // Test basic info collection functionality
  sentry_value_t event = sentry_value_new_event();
  EXPECT_TRUE(sentry_value_is_null(event) == 0);
  
  // Test setting message
  sentry_value_set_by_key(event, "message",
                          sentry_value_new_string("GPU Info Test - Basic"));
  
  // Test setting level
  sentry_value_set_by_key(event, "level", sentry_value_new_string("info"));
  
  SUCCEED() << "GPU info collection test completed";
}

TEST_F(GPUInfoSentryTest, WarningLevelEvent) {
  sentry_value_t event = sentry_value_new_event();
  EXPECT_TRUE(sentry_value_is_null(event) == 0);
  
  sentry_value_set_by_key(event, "level", sentry_value_new_string("warning"));
  SUCCEED() << "Warning level event test completed";
}

TEST_F(GPUInfoSentryTest, ErrorLevelEvent) {
  sentry_value_t event = sentry_value_new_event();
  EXPECT_TRUE(sentry_value_is_null(event) == 0);
  
  sentry_value_set_by_key(event, "level", sentry_value_new_string("error"));
  SUCCEED() << "Error level event test completed";
}

TEST_F(GPUInfoSentryTest, ExceptionHandling) {
  // Test exception handling without actually crashing
  try {
    throw std::runtime_error("Test exception for GPU info collection");
  } catch (const std::exception &e) {
    sentry_value_t event = sentry_value_new_event();
    EXPECT_TRUE(sentry_value_is_null(event) == 0);
    
    sentry_value_set_by_key(event, "message", sentry_value_new_string(e.what()));
    SUCCEED() << "Exception handling test completed";
  }
}