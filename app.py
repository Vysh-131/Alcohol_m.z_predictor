import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline  # <--- NEW: For smoothing curves
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
import sys

print("Loading models")

try:
    model_tics = joblib.load('TICS_Model.pkl')
    model_pics = joblib.load('PICS_Model.pkl')
    print("Models loaded.")
except FileNotFoundError:
    print(".pkl files not found.")
    sys.exit()

MASS_BINS = np.arange(1, 151)

def get_features(smiles, energy):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = mfgen.GetFingerprintAsNumPy(mol)
    return [energy, Descriptors.MolWt(mol)] + list(fp)

def main():
    while True:
        print("\n" + "="*50)
        print(" NEW ANALYSIS:")
        print("="*50)
        
        name = input("Molecule Name (or 'q' to quit): ").strip()
        if name.lower() == 'q': break
        smiles = input(f"Enter SMILES for {name}: ").strip()
        
        try:
            target_energy = float(input(" Target Energy (eV) for PICS (default 70): ").strip())
        except:
            target_energy = 70.0

        print(f"\n Analyzing {name}...")
        
        feats = get_features(smiles, target_energy)
        single_tics = model_tics.predict([feats])[0]
        single_pics_vec = model_pics.predict([feats])[0]
        
        print("   -> Generating smooth TICS curve...")
        energy_range = np.arange(10, 501, 10) 
        tics_curve = []
        
        for e in energy_range:
            f = get_features(smiles, e)
            val = model_tics.predict([f])[0]
            tics_curve.append(val)
            
        print("\n" + "-"*40)
        print(f" REPORT: {name.upper()}")
        print("-" * 40)
        print(f"🔹 TICS at {target_energy} eV: {single_tics:.4f} Å²")
        
    
        data_points = []
        for i, val in enumerate(single_pics_vec):
            if val > 0.01: data_points.append((MASS_BINS[i], val))
        data_points.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🔹 Top 3 PICS Fragments (at {target_energy} eV):")
        for m, v in data_points[:3]:
            print(f"   m/z {m:<4} | {v:.4f} Å²")
            
        plot_improved_lines(name, target_energy, single_pics_vec, energy_range, tics_curve)

def plot_improved_lines(name, target_e, pics_vec, e_range, tics_vals):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4)
    
    X_Y_Spline = make_interp_spline(e_range, tics_vals)
    X_smooth = np.linspace(e_range.min(), e_range.max(), 500)
    Y_smooth = X_Y_Spline(X_smooth)
    
    ax1.plot(X_smooth, Y_smooth, color='#2980b9', linewidth=2.5, label='TICS Trend')
    ax1.fill_between(X_smooth, Y_smooth, color='#2980b9', alpha=0.1)
    
    closest_e_idx = np.abs(e_range - target_e).argmin()
    ax1.plot(target_e, tics_vals[closest_e_idx], 'ro', markersize=8, 
             markeredgecolor='white', markeredgewidth=2, label=f"Selected ({target_e} eV)")
    
    ax1.set_title(f"Total Ionization Cross Section (TICS): {name}", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Electron Energy (eV)", fontsize=12)
    ax1.set_ylabel("Cross Section (Å²)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right')

    ax2.plot(MASS_BINS, pics_vec, color='#c0392b', linewidth=2, label='Fragment Signal')
    ax2.fill_between(MASS_BINS, pics_vec, color='#c0392b', alpha=0.1)
    
    peaks = [(MASS_BINS[i], val) for i, val in enumerate(pics_vec) if val > 0.05]
    peaks.sort(key=lambda x: x[1], reverse=True)

    for mass, val in peaks[:3]:
        ax2.annotate(f"{mass}", xy=(mass, val), xytext=(mass, val + 0.1),
                     ha='center', fontsize=10, fontweight='bold', color='#c0392b')

    ax2.set_title(f"Partial Cross Sections (PICS) Trace @ {target_e} eV", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Fragment Mass (m/z)", fontsize=12)
    ax2.set_ylabel("Cross Section (Å²)", fontsize=12)
    ax2.set_xlim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.6)

    print("\n Opening Enhanced Graphs...")
    plt.show()

if __name__ == "__main__":
    main()
