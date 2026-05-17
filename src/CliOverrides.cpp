#include "CliOverrides.hpp"

#include <stdexcept>
#include <string>
#include <unordered_set>

using json = nlohmann::json;

namespace {

// Every JSON key Config::fromJson accepts. Kept in sync with src/Config.cpp.
const std::unordered_set<std::string>& allowedKeys() {
    static const std::unordered_set<std::string> keys = {
        "N", "phi",
        "Pe", "De", "R", "C", "delta", "potential",
        "integrator", "max_drift", "r_skin",
        "t_end", "output_dt", "abs_tol", "rel_tol", "dt_init",
        "output_file", "init_mode", "seed",
        "compute_correlations", "corr_dt_max", "n_corr_steps", "t_warm",
        "compute_contact_durations",
    };
    return keys;
}

// json::parse("wca") throws; we want that to fall back to a string. json::parse
// also accepts bare numbers and booleans, which is exactly what we want for
// --N 500, --phi 0.7, --compute_correlations true.
json parseValue(const std::string& value) {
    try {
        return json::parse(value);
    } catch (const json::parse_error&) {
        return json(value);
    }
}

} // namespace

namespace CliOverrides {

json parse(int argc, char** argv, int start) {
    json out = json::object();

    for (int i = start; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg.size() < 3 || arg.substr(0, 2) != "--") {
            throw std::runtime_error(
                "CLI: expected a flag of the form --key value or --key=value, got '"
                + arg + "'");
        }

        std::string key;
        std::string value;
        const auto eq = arg.find('=');
        if (eq != std::string::npos) {
            key   = arg.substr(2, eq - 2);
            value = arg.substr(eq + 1);
        } else {
            key = arg.substr(2);
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    "CLI: flag '--" + key + "' is missing a value");
            }
            value = argv[++i];
        }

        if (key.empty()) {
            throw std::runtime_error("CLI: empty flag name in '" + arg + "'");
        }
        if (!allowedKeys().count(key)) {
            throw std::runtime_error(
                "CLI: unknown flag '--" + key +
                "'. Allowed flags match Config JSON keys "
                "(N, phi, Pe, De, R, C, delta, potential, integrator, "
                "max_drift, r_skin, t_end, output_dt, abs_tol, rel_tol, "
                "dt_init, output_file, init_mode, seed, "
                "compute_correlations, corr_dt_max, n_corr_steps, t_warm, "
                "compute_contact_durations).");
        }
        if (out.contains(key)) {
            throw std::runtime_error(
                "CLI: flag '--" + key + "' specified more than once");
        }

        out[key] = parseValue(value);
    }

    return out;
}

} // namespace CliOverrides
