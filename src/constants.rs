//! Physical constants (natural units, energies in GeV).

/// Non-reduced Planck mass `M_Pl = 1.22091 × 10^19 GeV`.
pub const M_PLANCK: f64 = 1.22091e19;

/// Present CMB temperature, 2.7255 K, in GeV.
pub const T_CMB: f64 = 2.7255 * 8.617_333_262e-14;

/// `ħc` in GeV cm, used to convert GeV³ to cm⁻³.
pub const HBAR_C_GEV_CM: f64 = 1.973_269_804e-14;

/// Critical density divided by `h²`, in GeV cm⁻³ (PDG 2024).
pub const RHO_CRIT_OVER_H2: f64 = 1.053_672e-5;

/// Planck 2018 cold dark matter density `Ω_c h² = 0.120`.
pub const OMEGA_DM_H2_PLANCK: f64 = 0.120;

/// Higgs mass \[GeV\] (value used in arXiv:1706.07433).
pub const M_HIGGS: f64 = 125.09;

/// Higgs vacuum expectation value \[GeV\].
pub const V_HIGGS: f64 = 246.2;

/// Temperature of the QCD transition \[GeV\] (used by the Scalar Singlet model).
pub const T_QCD: f64 = 0.154;
