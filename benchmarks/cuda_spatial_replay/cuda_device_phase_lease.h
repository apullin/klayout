/*
 * Optional crash-safe cross-process serialization for high-memory CUDA
 * transactions.  The lease covers device work only; callers keep compact
 * hierarchy validation and host lowering outside it whenever practical.
 */

#ifndef KLAYOUT_CUDA_DEVICE_PHASE_LEASE_H
#define KLAYOUT_CUDA_DEVICE_PHASE_LEASE_H

#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

namespace klayout_cuda {

class DevicePhaseLease
{
public:
  DevicePhaseLease(int device, const char *role)
      : m_role(role ? role : "unknown")
  {
    if (!enabled()) return;

    const char *configured_path =
        std::getenv("KLAYOUT_CUDA_DEVICE_LEASE_PATH");
    m_path =
        configured_path && *configured_path
            ? configured_path
            : "/tmp/klayout-cuda-spatial-device-" +
                  std::to_string(device) + ".lock";
    m_fd = open(
        m_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
    if (m_fd < 0) {
      throw std::runtime_error(
          "open CUDA device-phase lease " + m_path + ": " +
          std::to_string(errno));
    }

    const auto begin = Clock::now();
    const auto deadline = begin + wait_budget();
    for (;;) {
      if (flock(m_fd, LOCK_EX | LOCK_NB) == 0) break;
      const int error = errno;
      if (error != EWOULDBLOCK && error != EAGAIN) {
        close_fd();
        throw std::runtime_error(
            "acquire CUDA device-phase lease " + m_path + ": " +
            std::to_string(error));
      }
      const auto now = Clock::now();
      if (now >= deadline) {
        close_fd();
        throw std::runtime_error(
            "timed out waiting for CUDA device-phase lease " + m_path);
      }
      std::this_thread::sleep_for(
          std::min(
              deadline - now,
              std::chrono::duration_cast<Clock::duration>(
                  std::chrono::milliseconds(10))));
    }
    m_acquired = Clock::now();
    if (telemetry_enabled()) {
      std::fprintf(
          stderr,
          "KLAYOUT_CUDA_DEVICE_LEASE role=%s device=%d "
          "wait_ms=%.3f disposition=acquired\n",
          m_role.c_str(), device,
          milliseconds(begin, m_acquired));
    }
  }

  ~DevicePhaseLease()
  {
    if (m_fd < 0) return;
    const auto release = Clock::now();
    const int unlock_status = flock(m_fd, LOCK_UN);
    if (telemetry_enabled()) {
      std::fprintf(
          stderr,
          "KLAYOUT_CUDA_DEVICE_LEASE role=%s hold_ms=%.3f "
          "disposition=%s\n",
          m_role.c_str(), milliseconds(m_acquired, release),
          unlock_status == 0 ? "released" : "release-error");
    }
    close_fd();
  }

  DevicePhaseLease(const DevicePhaseLease &) = delete;
  DevicePhaseLease &operator=(const DevicePhaseLease &) = delete;

private:
  using Clock = std::chrono::steady_clock;

  static bool environment_enabled(const char *name)
  {
    const char *value = std::getenv(name);
    return value && *value &&
           !(value[0] == '0' && value[1] == '\0');
  }

  static bool enabled()
  {
    return environment_enabled("KLAYOUT_CUDA_DEVICE_LEASE");
  }

  static bool telemetry_enabled()
  {
    return environment_enabled(
        "KLAYOUT_CUDA_DEVICE_LEASE_TELEMETRY");
  }

  static std::chrono::milliseconds wait_budget()
  {
    constexpr unsigned long long kDefaultMilliseconds = 20000;
    constexpr unsigned long long kMaximumMilliseconds = 60000;
    const char *value =
        std::getenv("KLAYOUT_CUDA_DEVICE_LEASE_WAIT_MS");
    if (!value || !*value) {
      return std::chrono::milliseconds(kDefaultMilliseconds);
    }
    char *end = nullptr;
    const unsigned long long parsed =
        std::strtoull(value, &end, 10);
    if (end == value || !end || *end) {
      return std::chrono::milliseconds(kDefaultMilliseconds);
    }
    return std::chrono::milliseconds(
        std::min(parsed, kMaximumMilliseconds));
  }

  static double milliseconds(
      Clock::time_point begin, Clock::time_point end)
  {
    return std::chrono::duration<double, std::milli>(
               end - begin)
        .count();
  }

  void close_fd()
  {
    if (m_fd >= 0) {
      close(m_fd);
      m_fd = -1;
    }
  }

  int m_fd = -1;
  std::string m_path;
  std::string m_role;
  Clock::time_point m_acquired{};
};

}  // namespace klayout_cuda

#endif
