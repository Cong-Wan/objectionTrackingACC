/*
Author: wilbur
Version: 1.0
Date: 2026-06-27
Description: 全局日志接口。所有日志函数线程安全，由 --debug 控制是否输出 DEBUG 级别日志。
*/

#pragma once

#include <string>

void initLogger(bool debugEnabled);
bool isDebugEnabled();

void logInfo(const std::string& message);
void logDebug(const std::string& message);
void logWarn(const std::string& message);
void logError(const std::string& message);
