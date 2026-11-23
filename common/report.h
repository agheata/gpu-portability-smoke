// Result lines collected by run.sh:  RESULT <id> <PASS|FAIL|WARN|INFO> <detail>
//   FAIL: a required feature does not work.  WARN: an optional feature does not work.  INFO: a measurement.
#pragma once

#include <cstdarg>
#include <cstdio>

struct Report {
  int failures = 0;

  void result(const char *id, const char *status, const char *fmt, ...)
  {
    char detail[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(detail, sizeof(detail), fmt, ap);
    va_end(ap);
    std::printf("RESULT %s %s %s\n", id, status, detail);
    std::fflush(stdout);
  }

  /// Required check
  void check(const char *id, bool ok, const char *fmt = "", ...)
  {
    char detail[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(detail, sizeof(detail), fmt, ap);
    va_end(ap);
    if (!ok) ++failures;
    result(id, ok ? "PASS" : "FAIL", "%s", detail);
  }

  /// Optional check: a failure is reported as WARN
  void optional(const char *id, bool ok, const char *fmt = "", ...)
  {
    char detail[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(detail, sizeof(detail), fmt, ap);
    va_end(ap);
    result(id, ok ? "PASS" : "WARN", "%s", detail);
  }

  int finish() const { return failures == 0 ? 0 : 1; }
};
