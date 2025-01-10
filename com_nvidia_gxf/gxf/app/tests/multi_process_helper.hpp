/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <functional>
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <chrono>
#include <thread>
#include "common/logger.hpp"

using ProcessEntrance = std::function<int(const std::string&)>;

pid_t processRun(const ProcessEntrance& func, const std::string& arg) {
  pid_t pid = fork();
  if (pid == -1) {
    GXF_LOG_ERROR("Fork subprocess failed");
    exit(1);
  } else if (pid == 0) {
    // Child process
    int exitStatus = func(arg);
    _exit(exitStatus);
  }
  // Parent process returns the PID of the child
  return pid;
}

int processWait(pid_t pid) {
  int status;
  waitpid(pid, &status, 0);
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  // Return -1 if the child did not exit normally
  return -1;
}
