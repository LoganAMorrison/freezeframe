/// Standard-Model fermions that a mediator can couple to.
///
/// Masses are those used by DRAKE v1.0 (quark masses are running masses
/// relevant for scattering); `N_c` is the number of colours.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SmFermion {
    /// e
    Electron,
    /// μ
    Muon,
    /// τ
    Tau,
    /// u
    Up,
    /// d
    Down,
    /// s
    Strange,
    /// c
    Charm,
    /// b
    Bottom,
    /// t
    Top,
}

impl SmFermion {
    /// All SM fermions with a Higgs Yukawa coupling.
    pub const ALL: [SmFermion; 9] = [
        Self::Up,
        Self::Down,
        Self::Strange,
        Self::Charm,
        Self::Bottom,
        Self::Top,
        Self::Electron,
        Self::Muon,
        Self::Tau,
    ];

    /// Charged leptons.
    pub const LEPTONS: [SmFermion; 3] = [Self::Electron, Self::Muon, Self::Tau];

    /// Mass \[GeV\].
    pub fn mass(self) -> f64 {
        match self {
            Self::Electron => 0.000_511,
            Self::Muon => 0.105_66,
            Self::Tau => 1.777,
            Self::Up => 0.003,
            Self::Down => 0.006,
            Self::Strange => 0.103,
            Self::Charm => 1.874_331_282_034_321,
            Self::Bottom => 4.885_945_494_606_712,
            Self::Top => 172.7,
        }
    }

    /// Number of colours.
    pub fn colors(self) -> f64 {
        if self.is_quark() {
            3.0
        } else {
            1.0
        }
    }

    /// Whether this is a quark.
    pub fn is_quark(self) -> bool {
        !matches!(self, Self::Electron | Self::Muon | Self::Tau)
    }

    /// Whether this is one of the light quarks u, d, s.
    pub fn is_light_quark(self) -> bool {
        matches!(self, Self::Up | Self::Down | Self::Strange)
    }
}
