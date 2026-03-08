"""
=============================================================
 SATELLITE IMAGE LAND COVER CLASSIFICATION TOOL
 ISRO Bhuvan / Bhoonidhi Portal - LISS-IV / Resourcesat Data
=============================================================
Author: Built for ISRO Remote Sensing Project
Libraries: Rasterio, Scikit-learn, TensorFlow (optional), Folium
=============================================================
"""

# ─────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.plot import show
from rasterio.transform import from_bounds
import os
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Folium for interactive map
import folium
from folium import plugins
import json

print("✅ All core libraries imported successfully!")

# ─────────────────────────────────────────────
# SECTION 2: CONFIGURATION
# ─────────────────────────────────────────────

# Land Cover Classes (modify as needed)
LAND_COVER_CLASSES = {
    0: "Unclassified",
    1: "Urban / Built-up",
    2: "Forest / Dense Vegetation",
    3: "Agriculture / Cropland",
    4: "Water Bodies",
    5: "Barren / Rocky Land",
    6: "Wetlands / Marshy",
}

# Color map for visualization
CLASS_COLORS = {
    0: "#808080",   # Gray - Unclassified
    1: "#FF4444",   # Red - Urban
    2: "#228B22",   # Green - Forest
    3: "#ADFF2F",   # Yellow-Green - Agriculture
    4: "#1E90FF",   # Blue - Water
    5: "#D2B48C",   # Tan - Barren
    6: "#40E0D0",   # Turquoise - Wetlands
}

# ─────────────────────────────────────────────
# SECTION 3: SYNTHETIC DATA GENERATOR
# (Use this when you don't have real satellite data yet)
# ─────────────────────────────────────────────

def generate_synthetic_satellite_data(height=256, width=256, n_bands=4, seed=42):
    """
    Generates realistic synthetic multispectral satellite data.
    Bands: [Blue, Green, Red, NIR] - similar to LISS-IV / Resourcesat
    
    REPLACE THIS with real rasterio.open() when you have actual data.
    """
    np.random.seed(seed)
    
    print("\n📡 Generating synthetic satellite data (256x256, 4 bands)...")
    print("   ➤ Bands: Blue, Green, Red, NIR (like LISS-IV)")
    
    # Create base image
    image = np.zeros((n_bands, height, width), dtype=np.float32)
    
    # ── Urban areas (top-left quadrant)
    image[:, :80, :80] = np.random.normal(
        [0.25, 0.25, 0.30, 0.20], 0.05, (4, 80, 80)
    ).clip(0, 1).transpose(2,0,1)[:4,:,:]  # high red reflectance

    # ── Forest (top-right quadrant)
    image[:, :80, 80:] = np.random.normal(
        [0.05, 0.15, 0.08, 0.45], 0.03, (4, 80, 176)
    ).clip(0, 1).transpose(2,0,1)[:4,:,:]  # high NIR

    # ── Water (bottom-left)
    image[:, 160:, :100] = np.random.normal(
        [0.10, 0.08, 0.05, 0.02], 0.02, (4, 96, 100)
    ).clip(0, 1).transpose(2,0,1)[:4,:,:]  # low all bands

    # ── Agriculture (center)
    image[:, 80:160, 80:176] = np.random.normal(
        [0.08, 0.25, 0.12, 0.38], 0.04, (4, 80, 96)
    ).clip(0, 1).transpose(2,0,1)[:4,:,:]  # medium NIR

    # ── Barren land (bottom-right)
    image[:, 160:, 100:] = np.random.normal(
        [0.30, 0.32, 0.35, 0.28], 0.06, (4, 96, 156)
    ).clip(0, 1).transpose(2,0,1)[:4,:,:]  # uniform reflectance

    # ── Wetlands (middle strip)
    image[:, 80:160, :80] = np.random.normal(
        [0.12, 0.18, 0.10, 0.15], 0.03, (4, 80, 80)
    ).clip(0, 1).transpose(2,0,1)[:4,:,:]

    # Add realistic noise
    noise = np.random.normal(0, 0.02, image.shape)
    image = (image + noise).clip(0, 1)
    
    print(f"   ✅ Synthetic image shape: {image.shape}")
    return image


def generate_ground_truth_labels(height=256, width=256):
    """
    Creates ground truth labels matching the synthetic data layout.
    Replace with actual labeled data (shapefiles/GeoJSON) for real use.
    """
    labels = np.zeros((height, width), dtype=np.uint8)
    
    labels[:80, :80]     = 1   # Urban
    labels[:80, 80:]     = 2   # Forest
    labels[160:, :100]   = 4   # Water
    labels[80:160, 80:176] = 3  # Agriculture
    labels[160:, 100:]   = 5   # Barren
    labels[80:160, :80]  = 6   # Wetlands
    
    return labels


# ─────────────────────────────────────────────
# SECTION 4: REAL SATELLITE DATA LOADER
# (Use this when you have actual downloaded data from Bhuvan)
# ─────────────────────────────────────────────

def load_real_satellite_image(file_path):
    """
    Load a real satellite image from Bhuvan / Bhoonidhi portal.
    Supports GeoTIFF format (most common from ISRO portals).
    
    Args:
        file_path: Path to your downloaded .tif file
    
    Returns:
        image array (bands, height, width), metadata dict
    """
    print(f"\n📂 Loading real satellite image: {file_path}")
    
    with rasterio.open(file_path) as src:
        print(f"   ➤ CRS: {src.crs}")
        print(f"   ➤ Bands: {src.count}")
        print(f"   ➤ Resolution: {src.res}")
        print(f"   ➤ Bounds: {src.bounds}")
        print(f"   ➤ Size: {src.width} x {src.height}")
        
        image = src.read()  # Shape: (bands, height, width)
        meta = src.meta
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
    
    # Normalize to 0-1 range
    image = image.astype(np.float32)
    for b in range(image.shape[0]):
        band_min, band_max = image[b].min(), image[b].max()
        if band_max > band_min:
            image[b] = (image[b] - band_min) / (band_max - band_min)
    
    print(f"   ✅ Image loaded: shape={image.shape}, dtype={image.dtype}")
    return image, meta, bounds, crs


# ─────────────────────────────────────────────
# SECTION 5: FEATURE EXTRACTION
# ─────────────────────────────────────────────

def compute_spectral_indices(image):
    """
    Compute remote sensing indices.
    Assumes bands: [Blue=0, Green=1, Red=2, NIR=3]
    
    For LISS-IV:  Band2=Green, Band3=Red, Band4=NIR
    For Resourcesat: B2=Green, B3=Red, B4=NIR
    """
    eps = 1e-8
    
    blue  = image[0] if image.shape[0] > 3 else image[0]
    green = image[1] if image.shape[0] > 1 else image[0]
    red   = image[2] if image.shape[0] > 2 else image[0]
    nir   = image[3] if image.shape[0] > 3 else image[0]
    
    # NDVI - Normalized Difference Vegetation Index
    ndvi = (nir - red) / (nir + red + eps)
    
    # NDWI - Normalized Difference Water Index
    ndwi = (green - nir) / (green + nir + eps)
    
    # NDBI - Normalized Difference Built-up Index
    # Using SWIR approximation with red for LISS-IV (no SWIR band)
    ndbi = (red - nir) / (red + nir + eps)
    
    # EVI - Enhanced Vegetation Index
    evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + eps)
    
    print(f"\n📊 Spectral Indices Computed:")
    print(f"   NDVI  range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
    print(f"   NDWI  range: [{ndwi.min():.3f}, {ndwi.max():.3f}]")
    print(f"   NDBI  range: [{ndbi.min():.3f}, {ndbi.max():.3f}]")
    print(f"   EVI   range: [{evi.min():.3f},  {evi.max():.3f}]")
    
    return ndvi, ndwi, ndbi, evi


def extract_features(image):
    """
    Extract all features for ML classification.
    Returns feature matrix of shape (n_pixels, n_features)
    """
    n_bands, height, width = image.shape
    
    # Raw band values
    bands_flat = image.reshape(n_bands, -1).T  # (pixels, bands)
    
    # Spectral indices
    ndvi, ndwi, ndbi, evi = compute_spectral_indices(image)
    indices_flat = np.stack([
        ndvi.ravel(), ndwi.ravel(), ndbi.ravel(), evi.ravel()
    ], axis=1)  # (pixels, 4)
    
    # Band ratios (useful for ISRO data)
    eps = 1e-8
    red_green_ratio = (image[2] / (image[1] + eps)).ravel().reshape(-1,1)
    nir_red_ratio   = (image[3] / (image[2] + eps)).ravel().reshape(-1,1)
    
    # Combine all features
    features = np.concatenate([
        bands_flat,          # 4 raw bands
        indices_flat,        # 4 spectral indices
        red_green_ratio,     # 1 ratio
        nir_red_ratio,       # 1 ratio
    ], axis=1)
    
    print(f"\n🔢 Feature matrix shape: {features.shape}")
    print(f"   Features: 4 bands + NDVI + NDWI + NDBI + EVI + 2 ratios = 10 total")
    
    return features


# ─────────────────────────────────────────────
# SECTION 6: RANDOM FOREST CLASSIFIER
# ─────────────────────────────────────────────

def train_random_forest(features, labels_flat, class_dict=LAND_COVER_CLASSES):
    """
    Train a Random Forest classifier for land cover.
    """
    print("\n🌲 Training Random Forest Classifier...")
    
    # Remove unclassified pixels (label == 0)
    mask = labels_flat > 0
    X = features[mask]
    y = labels_flat[mask]
    
    print(f"   Training pixels: {X.shape[0]}")
    print(f"   Classes found: {np.unique(y)}")
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,           # Use all CPU cores
        class_weight='balanced'
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Training Complete!")
    print(f"   Overall Accuracy: {acc * 100:.2f}%")
    print(f"\n📋 Classification Report:")
    class_names = [class_dict.get(c, str(c)) for c in np.unique(y_test)]
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return rf_model, scaler, acc


def classify_image(model, scaler, features, height, width):
    """
    Apply trained model to classify the entire image.
    """
    print("\n🗺️  Classifying full image...")
    
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)
    
    # Reshape back to image
    classified_map = predictions.reshape(height, width)
    
    print(f"   ✅ Classification complete. Map shape: {classified_map.shape}")
    return classified_map


# ─────────────────────────────────────────────
# SECTION 7: VISUALIZATION
# ─────────────────────────────────────────────

def visualize_results(image, true_labels, predicted_map, class_dict=LAND_COVER_CLASSES, 
                       colors=CLASS_COLORS):
    """
    Create a 4-panel visualization of results.
    """
    print("\n🎨 Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor('#0D1117')
    
    for ax in axes.ravel():
        ax.set_facecolor('#0D1117')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
    
    # ── Panel 1: RGB Composite (Red, Green, Blue)
    ax1 = axes[0, 0]
    rgb = np.stack([image[2], image[1], image[0]], axis=2)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    ax1.imshow(rgb)
    ax1.set_title("🛰️  RGB Composite (Bands 3-2-1)", color='white', fontsize=13, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # ── Panel 2: False Color NIR (NIR, Red, Green)
    ax2 = axes[0, 1]
    nir_rgb = np.stack([image[3], image[2], image[1]], axis=2)
    nir_rgb = (nir_rgb - nir_rgb.min()) / (nir_rgb.max() - nir_rgb.min())
    ax2.imshow(nir_rgb)
    ax2.set_title("🌿 False Color NIR (Vegetation = Red)", color='white', fontsize=13, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # ── Panel 3: Ground Truth
    ax3 = axes[1, 0]
    cmap_vals = [colors.get(k, '#000000') for k in sorted(colors.keys())]
    
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(cmap_vals)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, len(class_dict)+0.5), ncolors=len(class_dict))
    
    ax3.imshow(true_labels, cmap=cmap, norm=norm, interpolation='nearest')
    ax3.set_title("📌 Ground Truth Labels", color='white', fontsize=13, fontweight='bold', pad=10)
    ax3.axis('off')
    
    # ── Panel 4: Predicted Classification
    ax4 = axes[1, 1]
    im = ax4.imshow(predicted_map, cmap=cmap, norm=norm, interpolation='nearest')
    ax4.set_title("🤖 ML Predicted Land Cover Map", color='white', fontsize=13, fontweight='bold', pad=10)
    ax4.axis('off')
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=colors[k], label=v)
        for k, v in class_dict.items() if k > 0
    ]
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=3,
        fontsize=11,
        framealpha=0.3,
        facecolor='#161B22',
        edgecolor='#444',
        labelcolor='white',
        bbox_to_anchor=(0.5, 0.01)
    )
    
    plt.suptitle(
        "🛰️  ISRO Satellite Image — Land Cover Classification\n(LISS-IV / Resourcesat | Random Forest ML)",
        color='white', fontsize=16, fontweight='bold', y=0.99
    )
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    plt.savefig("land_cover_classification.png", dpi=150, bbox_inches='tight', 
                facecolor='#0D1117')
    plt.show()
    print("   ✅ Saved: land_cover_classification.png")


def plot_feature_importance(model, scaler):
    """Plot feature importance from Random Forest."""
    feature_names = ['Blue', 'Green', 'Red', 'NIR', 
                      'NDVI', 'NDWI', 'NDBI', 'EVI',
                      'Red/Green', 'NIR/Red']
    importances = model.feature_importances_
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')
    
    sorted_idx = np.argsort(importances)
    colors_bar = plt.cm.viridis(np.linspace(0.3, 1, len(importances)))
    
    bars = ax.barh([feature_names[i] for i in sorted_idx], 
                    importances[sorted_idx], color=colors_bar)
    
    ax.set_xlabel("Feature Importance", color='white')
    ax.set_title("🌲 Random Forest — Feature Importances", color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.show()
    print("   ✅ Saved: feature_importance.png")


# ─────────────────────────────────────────────
# SECTION 8: FOLIUM INTERACTIVE MAP
# ─────────────────────────────────────────────

def create_folium_map(classified_map, 
                       center_lat=20.5937, center_lon=78.9629,  # India center
                       class_dict=LAND_COVER_CLASSES,
                       colors=CLASS_COLORS,
                       output_file="land_cover_map.html"):
    """
    Create an interactive Folium map showing land cover classification.
    Default center: India (adjust center_lat/lon to your study area).
    
    For real data from Bhuvan, pass the actual bounds from rasterio.
    """
    print(f"\n🗺️  Creating interactive Folium map...")
    
    # Build the Folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='CartoDB dark_matter'
    )
    
    # Add multiple tile layers
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB positron', name='CartoDB Light').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite Imagery'
    ).add_to(m)
    
    # ── Coverage area marker (since we're using synthetic data)
    folium.Marker(
        location=[center_lat, center_lon],
        popup=folium.Popup(
            "<b>🛰️ Study Area</b><br>ISRO Land Cover Classification<br>"
            f"Classes: {len(class_dict)-1}<br>"
            f"Image Size: {classified_map.shape[0]}x{classified_map.shape[1]} px",
            max_width=300
        ),
        tooltip="📍 Study Area Center",
        icon=folium.Icon(color='red', icon='satellite', prefix='fa')
    ).add_to(m)
    
    # ── Legend as HTML
    legend_html = """
    <div style="position: fixed; bottom: 30px; right: 30px; z-index: 1000;
                background-color: rgba(0,0,0,0.85); padding: 15px 20px;
                border-radius: 10px; border: 1px solid #444;
                font-family: monospace; min-width: 220px;">
        <h4 style="color: #00FF88; margin: 0 0 12px 0; font-size: 14px;">
            🛰️ Land Cover Classes
        </h4>
    """
    for cls_id, cls_name in class_dict.items():
        if cls_id == 0:
            continue
        color = colors.get(cls_id, '#888888')
        legend_html += f"""
        <div style="display: flex; align-items: center; margin: 6px 0;">
            <div style="width: 18px; height: 18px; background: {color}; 
                        margin-right: 10px; border-radius: 3px; flex-shrink: 0;"></div>
            <span style="color: white; font-size: 12px;">{cls_name}</span>
        </div>"""
    
    # Class statistics
    unique, counts = np.unique(classified_map, return_counts=True)
    total = classified_map.size
    legend_html += "<hr style='border-color: #444; margin: 10px 0;'>"
    legend_html += "<p style='color: #AAA; font-size: 11px; margin: 5px 0;'>📊 Coverage %</p>"
    for u, c in zip(unique, counts):
        if u == 0:
            continue
        pct = (c / total) * 100
        name = class_dict.get(u, str(u))
        color = colors.get(u, '#888')
        legend_html += f"<p style='color: white; font-size: 11px; margin: 3px 0;'><span style='color:{color}'>■</span> {name[:15]}: {pct:.1f}%</p>"
    
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # ── Title
    title_html = """
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: rgba(0,0,0,0.8); padding: 10px 25px;
                border-radius: 25px; border: 1px solid #00FF88;
                font-family: monospace; text-align: center;">
        <span style="color: #00FF88; font-size: 16px; font-weight: bold;">
            🛰️ ISRO Satellite Land Cover Classification
        </span>
        <br>
        <span style="color: #AAA; font-size: 11px;">
            LISS-IV / Resourcesat | Random Forest ML | Bhuvan Portal
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Layer control
    folium.LayerControl().add_to(m)
    
    # Save
    m.save(output_file)
    print(f"   ✅ Interactive map saved: {output_file}")
    print(f"   🌐 Open in browser: file://{os.path.abspath(output_file)}")
    
    return m


# ─────────────────────────────────────────────
# SECTION 9: SAVE / LOAD MODEL
# ─────────────────────────────────────────────

def save_model(model, scaler, filepath="land_cover_rf_model.pkl"):
    joblib.dump({'model': model, 'scaler': scaler}, filepath)
    print(f"\n💾 Model saved: {filepath}")


def load_model(filepath="land_cover_rf_model.pkl"):
    data = joblib.load(filepath)
    print(f"📂 Model loaded: {filepath}")
    return data['model'], data['scaler']


# ─────────────────────────────────────────────
# SECTION 10: MAIN PIPELINE
# ─────────────────────────────────────────────

def run_full_pipeline(satellite_image_path=None):
    """
    Full end-to-end pipeline.
    
    Args:
        satellite_image_path: Path to your real .tif file from Bhuvan.
                              If None, uses synthetic data for demo.
    """
    print("=" * 60)
    print("  🛰️  ISRO LAND COVER CLASSIFICATION PIPELINE")
    print("=" * 60)
    
    # ── Step 1: Load Data
    if satellite_image_path and os.path.exists(satellite_image_path):
        print(f"\n📂 Using REAL satellite data: {satellite_image_path}")
        image, meta, bounds, crs = load_real_satellite_image(satellite_image_path)
        height, width = image.shape[1], image.shape[2]
        
        # You'll need real labels here (from shapefiles/manual labeling)
        # For now, generate synthetic labels for demonstration
        print("⚠️  Using synthetic labels (replace with your GIS labels)")
        true_labels = generate_ground_truth_labels(height, width)
        
        # Get map center from bounds
        center_lat = (bounds.top + bounds.bottom) / 2
        center_lon = (bounds.left + bounds.right) / 2
    else:
        print("\n🧪 Using SYNTHETIC data (demo mode)")
        print("   → Replace with real Bhuvan/Bhoonidhi data when available")
        image = generate_synthetic_satellite_data()
        height, width = image.shape[1], image.shape[2]
        true_labels = generate_ground_truth_labels(height, width)
        center_lat, center_lon = 20.5937, 78.9629  # India center
    
    # ── Step 2: Feature Extraction
    features = extract_features(image)
    labels_flat = true_labels.ravel()
    
    # ── Step 3: Train Classifier
    model, scaler, accuracy = train_random_forest(features, labels_flat)
    
    # ── Step 4: Classify Full Image
    predicted_map = classify_image(model, scaler, features, height, width)
    
    # ── Step 5: Visualize
    visualize_results(image, true_labels, predicted_map)
    plot_feature_importance(model, scaler)
    
    # ── Step 6: Create Folium Map
    folium_map = create_folium_map(
        predicted_map,
        center_lat=center_lat,
        center_lon=center_lon
    )
    
    # ── Step 7: Save Model
    save_model(model, scaler)
    
    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print("\n📁 Output Files:")
    print("   • land_cover_classification.png  → Classification map")
    print("   • feature_importance.png          → Feature importance chart")
    print("   • land_cover_map.html             → Interactive Folium map")
    print("   • land_cover_rf_model.pkl         → Saved ML model")
    print(f"\n   📊 Final Accuracy: {accuracy*100:.2f}%")
    
    return model, scaler, predicted_map, folium_map


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    
    # ── OPTION A: Demo with synthetic data (no download needed)
    model, scaler, predicted_map, folium_map = run_full_pipeline()
    
    # ── OPTION B: Use your real satellite data from Bhuvan
    # Download .tif from https://bhuvan.nrsc.gov.in or https://bhoonidhi.nrsc.gov.in
    # Then uncomment and set your file path:
    #
    # model, scaler, predicted_map, folium_map = run_full_pipeline(
    #     satellite_image_path="path/to/your/LISS-IV_image.tif"
    # )
