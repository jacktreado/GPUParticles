#include "Config.hpp"
#include "ForceCalculator.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

// =============================================================================
// Config unit tests
// -----------------------------------------------------------------------------
// New parameterization: working units (sigma, epsilon, gamma) are frozen at 1
// and the user supplies six dimensionless inputs (Pe, De, R, C, delta) plus
// N, phi, and the potential type. Microscopic params (f0, tau_theta, gamma_a,
// k_a, kT, L) are derived by Config::recompute() from those inputs.
// =============================================================================

class ConfigTest : public ::testing::Test {
protected:
    std::filesystem::path tempPath;

    void SetUp() override {
        tempPath = std::filesystem::temp_directory_path() / "bd_test_config.json";
    }
    void TearDown() override {
        std::filesystem::remove(tempPath);
    }

    void writeJson(const std::string& content) const {
        std::ofstream f(tempPath);
        f << content;
    }

    static std::string base(const std::string& extras = "") {
        return R"({"N":64,"phi":0.4)" + (extras.empty() ? "" : "," + extras) + "}";
    }
};

// ---- Required fields parsed correctly ---------------------------------------

TEST_F(ConfigTest, RequiredFieldsParsed) {
    writeJson(R"({"N":128,"phi":0.5})");
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_EQ(cfg.N, std::size_t{128});
    EXPECT_DOUBLE_EQ(cfg.phi, 0.5);
}

// ---- Frozen working units ---------------------------------------------------

TEST_F(ConfigTest, WorkingUnitsAreFrozenAtOne) {
    writeJson(base());
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_DOUBLE_EQ(cfg.sigma,   1.0);
    EXPECT_DOUBLE_EQ(cfg.epsilon, 1.0);
    EXPECT_DOUBLE_EQ(cfg.gamma,   1.0);
}

// ---- Dimensionless input defaults -------------------------------------------

TEST_F(ConfigTest, DefaultPe)        { writeJson(base()); EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).Pe,    1.0); }
TEST_F(ConfigTest, DefaultDe)        { writeJson(base()); EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).De,    1.0); }
TEST_F(ConfigTest, DefaultR)         { writeJson(base()); EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).R,     1.0); }
TEST_F(ConfigTest, DefaultC)         { writeJson(base()); EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).C,     1e-3); }
TEST_F(ConfigTest, DefaultDelta)     { writeJson(base()); EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).delta, 1.0); }

TEST_F(ConfigTest, DefaultPotentialIsWCA) {
    writeJson(base());
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_EQ(cfg.potential, PotentialType::WCA);
    EXPECT_EQ(cfg.potentialName(), "WCA");
}

// ---- Override dimensionless inputs ------------------------------------------

TEST_F(ConfigTest, OverridePe)    { writeJson(base(R"("Pe":12.5)"));    EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).Pe,    12.5); }
TEST_F(ConfigTest, OverrideDe)    { writeJson(base(R"("De":0.25)"));    EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).De,    0.25); }
TEST_F(ConfigTest, OverrideR)     { writeJson(base(R"("R":2.5)"));      EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).R,     2.5); }
TEST_F(ConfigTest, OverrideC)     { writeJson(base(R"("C":0.05)"));     EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).C,     0.05); }
TEST_F(ConfigTest, OverrideDelta) { writeJson(base(R"("delta":0.85)")); EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).delta, 0.85); }

TEST_F(ConfigTest, OverridePotentialSoftSphere) {
    writeJson(base(R"("potential":"soft_sphere")"));
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_EQ(cfg.potential, PotentialType::SoftSphere);
    EXPECT_EQ(cfg.potentialName(), "soft_sphere");
}

// Aliases tolerated by the parser.
TEST_F(ConfigTest, OverridePotentialSoftSphereAliases) {
    for (const char* name : {"softsphere", "SoftSphere", "soft-sphere"}) {
        writeJson(base(std::string(R"("potential":")") + name + R"(")"));
        Config cfg = Config::fromFile(tempPath.string());
        EXPECT_EQ(cfg.potential, PotentialType::SoftSphere) << "alias: " << name;
    }
}

// ---- Adaptive integrator + I/O defaults / overrides ------------------------

TEST_F(ConfigTest, DefaultTEnd)       { writeJson(base()); EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).t_end,     1.0); }
TEST_F(ConfigTest, DefaultOutputDt)   { writeJson(base()); EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).output_dt, 0.01); }
TEST_F(ConfigTest, DefaultAbsTol)     { writeJson(base()); EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).abs_tol,   1.0e-3); }
TEST_F(ConfigTest, DefaultRelTol)     { writeJson(base()); EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).rel_tol,   1.0e-2); }
TEST_F(ConfigTest, DefaultDtInit)     { writeJson(base()); EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).dt_init,   1.0e-3); }
TEST_F(ConfigTest, DefaultOutputFile) { writeJson(base()); EXPECT_EQ      (Config::fromFile(tempPath.string()).output_file, std::string{"trajectory.h5"}); }
TEST_F(ConfigTest, DefaultInitMode)   { writeJson(base()); EXPECT_EQ      (Config::fromFile(tempPath.string()).init_mode,   std::string{"lattice"}); }
TEST_F(ConfigTest, DefaultSeed)       { writeJson(base()); EXPECT_EQ      (Config::fromFile(tempPath.string()).seed,        std::uint64_t{12345}); }

TEST_F(ConfigTest, OverrideTEnd)            { writeJson(base(R"("t_end":5.0)"));        EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).t_end,     5.0); }
TEST_F(ConfigTest, OverrideOutputDt)        { writeJson(base(R"("output_dt":0.05)"));   EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).output_dt, 0.05); }
TEST_F(ConfigTest, OverrideAbsTol)          { writeJson(base(R"("abs_tol":1e-6)"));     EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).abs_tol,   1e-6); }
TEST_F(ConfigTest, OverrideRelTol)          { writeJson(base(R"("rel_tol":1e-5)"));     EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).rel_tol,   1e-5); }
TEST_F(ConfigTest, OverrideDtInit)          { writeJson(base(R"("dt_init":1e-4)"));     EXPECT_DOUBLE_EQ(Config::fromFile(tempPath.string()).dt_init,   1e-4); }
TEST_F(ConfigTest, OverrideInitModeRandom)  { writeJson(base(R"("init_mode":"random")")); EXPECT_EQ    (Config::fromFile(tempPath.string()).init_mode,  "random"); }
TEST_F(ConfigTest, OverrideSeed)            { writeJson(base(R"("seed":99999)"));         EXPECT_EQ    (Config::fromFile(tempPath.string()).seed,       std::uint64_t{99999}); }

// ---- Derived geometry --------------------------------------------------------

TEST_F(ConfigTest, BoxSizeDerivedFromNPhiSigma) {
    writeJson(R"({"N":64,"phi":0.4})");
    Config cfg = Config::fromFile(tempPath.string());
    const double pi = std::acos(-1.0);
    const double L_expected = std::sqrt(64.0 * pi * 1.0 * 1.0 / (4.0 * 0.4));
    EXPECT_NEAR(cfg.L, L_expected, 1e-12);
}

TEST_F(ConfigTest, RecomputeUpdatesBoxSize) {
    writeJson(R"({"N":256,"phi":0.4})");
    Config cfg = Config::fromFile(tempPath.string());
    const double L_original = cfg.L;
    cfg.N = 512;
    cfg.recompute();
    EXPECT_GT(cfg.L, L_original);
}

// ---- Derived microscopic parameters -----------------------------------------

TEST_F(ConfigTest, DerivedKTEqualsCEpsilon) {
    writeJson(base(R"("C":0.05)"));
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_DOUBLE_EQ(cfg.kT, 0.05 * cfg.epsilon);
}

TEST_F(ConfigTest, DerivedDiffusionCoefficient) {
    // D = kT / gamma; with gamma = 1, D = kT = C.
    writeJson(base(R"("C":0.5)"));
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_DOUBLE_EQ(cfg.diffusionCoefficient(), 0.5);
}

// SoftSphere has a closed-form f0 = (eps/sigma) * (1 - delta). With eps=sigma=1,
// f0 = 1 - delta.
TEST_F(ConfigTest, DerivedF0SoftSphere) {
    writeJson(base(R"("potential":"soft_sphere","delta":0.7)"));
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_DOUBLE_EQ(cfg.f0, 0.3);
}

// WCA at delta = 1 gives f0 = 24 eps/sigma.
TEST_F(ConfigTest, DerivedF0WCA) {
    writeJson(base(R"("potential":"WCA","delta":1.0)"));
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_DOUBLE_EQ(cfg.f0, 24.0);
}

// tau_theta = Pe * gamma * sigma / f0  (gamma=sigma=1 → Pe / f0).
TEST_F(ConfigTest, DerivedTauThetaFromPe) {
    writeJson(base(R"("potential":"soft_sphere","delta":0.5,"Pe":4.0)"));
    Config cfg = Config::fromFile(tempPath.string());
    // f0 = 0.5, tau_theta = 4.0 / 0.5 = 8.0.
    EXPECT_DOUBLE_EQ(cfg.f0, 0.5);
    EXPECT_DOUBLE_EQ(cfg.tau_theta, 8.0);
}

// gamma_a = R * gamma. With gamma=1, gamma_a = R.
TEST_F(ConfigTest, DerivedGammaAFromR) {
    writeJson(base(R"("R":2.5)"));
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_DOUBLE_EQ(cfg.gamma_a, 2.5);
}

// k_a = gamma_a / (De * tau_theta).
TEST_F(ConfigTest, DerivedKaFromDe) {
    writeJson(base(R"("potential":"soft_sphere","delta":0.5,"Pe":2.0,"R":3.0,"De":4.0)"));
    Config cfg = Config::fromFile(tempPath.string());
    // f0 = 0.5, tau_theta = 2.0/0.5 = 4.0, gamma_a = 3.0,
    // k_a = 3.0 / (4.0 * 4.0) = 3/16
    EXPECT_DOUBLE_EQ(cfg.tau_theta, 4.0);
    EXPECT_DOUBLE_EQ(cfg.gamma_a,   3.0);
    EXPECT_DOUBLE_EQ(cfg.k_a,       3.0 / 16.0);
}

// ---- Edge cases: passive / degenerate ---------------------------------------

// Delta beyond cutoff means f0 = 0 (passive); tau_theta and k_a become 0 too.
TEST_F(ConfigTest, PassiveLimitDeltaBeyondCutoffSoftSphere) {
    writeJson(base(R"("potential":"soft_sphere","delta":1.5,"Pe":10.0,"De":1.0,"R":1.0)"));
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_DOUBLE_EQ(cfg.f0,        0.0);
    EXPECT_DOUBLE_EQ(cfg.tau_theta, 0.0);
    EXPECT_DOUBLE_EQ(cfg.k_a,       0.0);
}

TEST_F(ConfigTest, PassiveLimitDeltaBeyondCutoffWCA) {
    writeJson(base(R"("potential":"WCA","delta":1.5,"Pe":10.0)"));
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_DOUBLE_EQ(cfg.f0,        0.0);
    EXPECT_DOUBLE_EQ(cfg.tau_theta, 0.0);
    EXPECT_DOUBLE_EQ(cfg.k_a,       0.0);
}

// Pe = 0: no rotational diffusion; k_a degenerates to 0.
TEST_F(ConfigTest, ZeroPeGivesZeroTauThetaAndKa) {
    writeJson(base(R"("potential":"soft_sphere","delta":0.5,"Pe":0.0)"));
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_DOUBLE_EQ(cfg.f0,        0.5);     // active drive still set
    EXPECT_DOUBLE_EQ(cfg.tau_theta, 0.0);
    EXPECT_DOUBLE_EQ(cfg.k_a,       0.0);
}

// De = 0: anchor decoupled (k_a = 0). gamma_a still set.
TEST_F(ConfigTest, ZeroDeGivesZeroKa) {
    writeJson(base(R"("potential":"soft_sphere","delta":0.5,"Pe":1.0,"De":0.0,"R":2.0)"));
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_DOUBLE_EQ(cfg.gamma_a, 2.0);
    EXPECT_DOUBLE_EQ(cfg.k_a,     0.0);
}

// R = 0: gamma_a = 0 (anchor frozen); k_a follows to 0 (no spring without friction).
TEST_F(ConfigTest, ZeroRGivesFrozenAnchor) {
    writeJson(base(R"("potential":"soft_sphere","delta":0.5,"Pe":1.0,"De":1.0,"R":0.0)"));
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_DOUBLE_EQ(cfg.gamma_a, 0.0);
    EXPECT_DOUBLE_EQ(cfg.k_a,     0.0);
}

// ---- Missing required fields throw -----------------------------------------

TEST_F(ConfigTest, MissingNThrows) {
    writeJson(R"({"phi":0.4})");
    EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error);
}

TEST_F(ConfigTest, MissingPhiThrows) {
    writeJson(R"({"N":64})");
    EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error);
}

// ---- Validation errors ------------------------------------------------------

TEST_F(ConfigTest, ZeroNThrows)         { writeJson(R"({"N":0,"phi":0.4})");          EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, PhiZeroThrows)       { writeJson(R"({"N":64,"phi":0.0})");         EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, PhiOneThrows)        { writeJson(R"({"N":64,"phi":1.0})");         EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, NegativePeThrows)    { writeJson(base(R"("Pe":-1.0)"));            EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, NegativeDeThrows)    { writeJson(base(R"("De":-1.0)"));            EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, NegativeRThrows)     { writeJson(base(R"("R":-1.0)"));             EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, NegativeCThrows)     { writeJson(base(R"("C":-1.0)"));             EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, ZeroDeltaThrows)     { writeJson(base(R"("delta":0.0)"));          EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, NegativeDeltaThrows) { writeJson(base(R"("delta":-0.5)"));         EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, NonPositiveTEndThrows)    { writeJson(base(R"("t_end":0.0)"));     EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, NonPositiveOutputDtThrows){ writeJson(base(R"("output_dt":-0.1)")); EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, NonPositiveAbsTolThrows)  { writeJson(base(R"("abs_tol":0.0)"));   EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, NonPositiveRelTolThrows)  { writeJson(base(R"("rel_tol":-1e-3)")); EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, NonPositiveDtInitThrows)  { writeJson(base(R"("dt_init":0.0)"));   EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, InvalidInitModeThrows) { writeJson(base(R"("init_mode":"hex")"));  EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error); }
TEST_F(ConfigTest, InvalidPotentialThrows) {
    writeJson(base(R"("potential":"lennard_jones")"));
    EXPECT_THROW(Config::fromFile(tempPath.string()), std::runtime_error);
}

TEST_F(ConfigTest, NonexistentFileThrows) {
    EXPECT_THROW(Config::fromFile("/nonexistent/path/bd_test.json"), std::runtime_error);
}

// Athermal limit (C = 0) is allowed and gives kT = 0, D = 0.
TEST_F(ConfigTest, AthermalLimitAccepted) {
    writeJson(base(R"("C":0.0)"));
    Config cfg = Config::fromFile(tempPath.string());
    EXPECT_DOUBLE_EQ(cfg.kT, 0.0);
    EXPECT_DOUBLE_EQ(cfg.diffusionCoefficient(), 0.0);
}
