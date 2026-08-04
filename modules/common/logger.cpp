/*
Author: wilbur
Version: 1.0
Date: 2026-06-27
Description: 全局日志实现。线程安全：内部用互斥量串行化每条日志，保证多线程输出不交错。
*/

#include "logger.h"

#include <iostream>
#include <mutex>

namespace {
std::mutex gLogMutex;
bool gDebugEnabled = false;
}

void initLogger(bool debugEnabled) {
    gDebugEnabled = debugEnabled;
}

bool isDebugEnabled() {
    return gDebugEnabled;
}

void logInfo(const std::string& message) {
    std::lock_guard<std::mutex> lock(gLogMutex);
    std::cout << "[INFO] " << message << std::endl;
}

void logDebug(const std::string& message) {
    if (!gDebugEnabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(gLogMutex);
    std::cout << "[DEBUG] " << message << std::endl;
}

void logWarn(const std::string& message) {
    std::lock_guard<std::mutex> lock(gLogMutex);
    std::cout << "[WARN] " << message << std::endl;
}

void logError(const std::string& message) {
    std::lock_guard<std::mutex> lock(gLogMutex);
    std::cerr << "[ERROR] " << message << std::endl;
}
