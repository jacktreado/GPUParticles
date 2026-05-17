#pragma once

#include <nlohmann/json.hpp>

// =============================================================================
// CliOverrides
// -----------------------------------------------------------------------------
// Parses `--key value` and `--key=value` CLI arguments into a JSON object that
// can be merged over the input.json before constructing a Config. Flag names
// must match a JSON field accepted by Config::fromJson; unknown names error
// out so typos can't silently produce a misconfigured run.
//
// Per-value types are auto-detected via nlohmann::json::parse: numbers, bools,
// and JSON literals come through as their native types; anything that fails
// to parse as JSON is kept as a string (e.g. "wca", "trajectory.h5").
// =============================================================================
namespace CliOverrides {

// Parse argv[start..argc) into a JSON object of overrides. Throws
// std::runtime_error on malformed args, unknown flag names, or duplicates.
nlohmann::json parse(int argc, char** argv, int start);

} // namespace CliOverrides
