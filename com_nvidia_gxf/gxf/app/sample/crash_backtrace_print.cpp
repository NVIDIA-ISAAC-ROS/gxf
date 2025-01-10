/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <csetjmp>
#include <cstring>
#include <iostream>
#include <string>

#include "gtest/gtest.h"

#include "common/assert.hpp"
#include "gxf/app/application.hpp"
#include "gxf/app/graph_entity.hpp"
#include "gxf/sample/ping_rx.hpp"
#include "gxf/sample/ping_tx.hpp"

namespace nvidia {
namespace gxf {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

class TestCrashApp : public Application {
 public:
  void compose() override {
    // create a codelet to generate 10 messages
    auto tx_entity = makeEntity<PingTx>("Tx",
                        makeTerm<CountSchedulingTerm>("count", Arg("count", 10)));

    // create a codelet to receive the messages
    auto rx_entity = makeEntity<PingRx>("Rx");

    // add data flow connection tx -> rx
    connect(tx_entity, rx_entity);

    // configure the scheduler component
    setScheduler(SchedulerType::kGreedy);

    make_crash();
  }

  Expected<void> make_crash() {
    GXF_LOG_INFO("Intentionally cause a crash by dereference null pointer");
    int* p = nullptr;
    *p = 42;  // Dereference null pointer to cause a crash
    return Success;
  }
};


TEST(TestApp, PrintCrashBacktrace) {
  int pipefd[2];
  ASSERT_EQ(pipe(pipefd), 0) << "Failed to create pipe";

  pid_t pid = fork();

  if (pid == 0) {
    // Child process
    close(pipefd[0]);  // Close unused read end
    dup2(pipefd[1], STDERR_FILENO);  // Redirect stderr to pipe
    close(pipefd[1]);

    auto app = create_app<TestCrashApp>();
    GXF_ASSERT_SUCCESS(app->setSeverity(GXF_SEVERITY_DEBUG));
    GXF_ASSERT_TRUE(app->loadExtensionManifest(kGxeManifestFilename).has_value());
    app->compose();
    app->run();
    exit(0);  // Should not reach here
  } else if (pid > 0) {
    // Parent process
    close(pipefd[1]);  // Close unused write end

    int status;
    waitpid(pid, &status, 0);

    if (WIFSIGNALED(status)) {
      int signal = WTERMSIG(status);
      std::cout << "Child process terminated by signal: " << strsignal(signal) << std::endl;

      // Check if it's a segmentation fault
      EXPECT_EQ(signal, SIGSEGV);

      // Read and check the output
      char buffer[4096];
      std::string output;
      ssize_t bytes_read;
      while ((bytes_read = read(pipefd[0], buffer, sizeof(buffer))) > 0) {
        output.append(buffer, bytes_read);
      }
      close(pipefd[0]);

      // Print the captured output to the terminal
      std::cout << "Captured output from child process:" << std::endl;
      std::cout << "\n----- BEGIN CAPTURED OUTPUT -----\n";
      std::cout << output << std::endl;
      std::cout << "----- END CAPTURED OUTPUT -----\n\n";

      // Check for backtrace in the output
      EXPECT_TRUE(output.find("==== backtrace (tid:") != std::string::npos)
        << "Backtrace header not found in output";

      // Check for specific function names or addresses in the backtrace
      EXPECT_TRUE(output.find("TestCrashApp::make_crash") != std::string::npos ||
                  output.find("crash_backtrace_print") != std::string::npos)
        << "Expected function not found in backtrace";

      // could add more specific checks here based on what expect to see in the backtrace
    } else {
      FAIL() << "Child process did not terminate by a signal as expected";
    }
  } else {
    FAIL() << "Fork failed: " << strerror(errno);
  }
}

}  // namespace gxf
}  // namespace nvidia
