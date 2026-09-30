"""
material_classifier.py
======================
Rule-based material composition classifier and hazard detection system.
This module extends the YOLOv8 detection with:
- Material composition inference
- Toxicity/hazard classification
- Economic recovery value estimation
- Recycling routing recommendations

Can be used standalone or imported by predict.py.

Usage:
    python material_classifier.py --class-name circuit_board
    python material_classifier.py --from-json predictions.json
    python material_classifier.py --all-classes
"""

import json
import argparse
import sys

# ─── MATERIAL DATABASE ────────────────────────────────────────────────────────

# Comprehensive material profiles based on PCB recycling literature
# Sources: Shafiee et al. (2019), Firsching et al. (2024)

MATERIAL_PROFILES = {
    "circuit_board": {
        "common_name": "Printed Circuit Board (PCB)",
        "materials_pct": {
            "copper": 15.0,
            "gold": 0.03,
            "silver": 0.15,
            "palladium": 0.01,
            "lead": 2.0,
            "tin": 4.0,
            "fibreglass_epoxy": 40.0,
            "plastic": 20.0,
            "iron": 4.0,
        },
        "hazardous_substances": {
            "lead": "Neurotoxin, damages kidneys and nervous system",
            "brominated_flame_retardants": "Persistent organic pollutant",
            "antimony": "Potential carcinogen",
        },
        "recovery_priority": "HIGH",
        "recovery_method": "Hydrometallurgical process / Smelting",
        "recycling_note": "High-value PCBs should be separated for precious metal recovery",
    },
    "mobile_phone": {
        "common_name": "Smartphone / Mobile Phone",
        "materials_pct": {
            "gold": 0.034,
            "silver": 0.35,
            "copper": 15.0,
            "cobalt": 5.0,
            "lithium": 2.0,
            "plastic": 40.0,
            "glass": 10.0,
            "aluminium": 8.0,
        },
        "hazardous_substances": {
            "lithium": "Fire/explosion risk if punctured",
            "cobalt": "Toxic if inhaled (dust form)",
        },
        "recovery_priority": "HIGH",
        "recovery_method": "Manual dismantling → battery removal → PCB processing",
        "recycling_note": "Remove battery before processing; high gold content in PCB",
    },
    "battery": {
        "common_name": "Battery (Li-ion / Lead-Acid / NiMH)",
        "materials_pct": {
            "lithium": 8.0,
            "cobalt": 15.0,
            "nickel": 10.0,
            "manganese": 5.0,
            "graphite": 15.0,
            "plastic_case": 15.0,
            "electrolyte": 10.0,
            "aluminium_foil": 8.0,
            "copper_foil": 5.0,
        },
        "hazardous_substances": {
            "lithium": "CRITICAL: Fire risk, thermal runaway possible",
            "cobalt": "Toxic if inhaled",
            "electrolyte_LiPF6": "Corrosive, toxic fluoride compounds",
            "lead": "(Lead-acid type) Severe neurotoxin",
            "cadmium": "(NiCd type) Carcinogen",
        },
        "recovery_priority": "CRITICAL_HAZARD",
        "recovery_method": "Dedicated battery recycling facility ONLY",
        "recycling_note": "⚠️ NEVER puncture, crush, or incinerate. Special handling required.",
    },
    "cable": {
        "common_name": "Cable / Wire",
        "materials_pct": {
            "copper": 60.0,
            "pvc_plastic": 30.0,
            "tin_coating": 2.0,
            "steel_shield": 5.0,
        },
        "hazardous_substances": {
            "pvc_plasticizers": "Phthalates - endocrine disruptors when burned",
        },
        "recovery_priority": "MEDIUM",
        "recovery_method": "Cable stripping → copper recovery",
        "recycling_note": "High copper content. Do not burn PVC insulation.",
    },
    "charger": {
        "common_name": "Phone/Laptop Charger",
        "materials_pct": {
            "copper": 25.0,
            "iron": 10.0,
            "plastic": 40.0,
            "aluminium": 8.0,
            "tin": 3.0,
        },
        "hazardous_substances": {
            "lead_solder": "Lead in solder joints",
        },
        "recovery_priority": "MEDIUM",
        "recovery_method": "Shredding → magnetic separation → copper recovery",
        "recycling_note": "Contains transformer with copper windings",
    },
    "keyboard": {
        "common_name": "Computer Keyboard",
        "materials_pct": {
            "plastic_abs": 70.0,
            "copper": 8.0,
            "steel": 5.0,
            "rubber": 10.0,
        },
        "hazardous_substances": {},
        "recovery_priority": "LOW",
        "recovery_method": "Shredding → plastic/metal separation",
        "recycling_note": "Mostly plastic. Low economic value per unit.",
    },
    "monitor": {
        "common_name": "Computer Monitor / Display",
        "materials_pct": {
            "glass": 50.0,
            "plastic": 25.0,
            "copper": 5.0,
            "aluminium": 8.0,
            "rare_earth_phosphors": 0.5,
            "steel": 5.0,
        },
        "hazardous_substances": {
            "mercury_backlight": "(CCFL/older LCD) Toxic to aquatic ecosystems",
            "lead_glass": "(CRT monitors) Contains significant lead",
        },
        "recovery_priority": "HIGH",
        "recovery_method": "CRT: specialist glass processing. LCD: mercury lamp removal first.",
        "recycling_note": "CRT monitors require specialist recycling due to lead glass.",
    },
    "mouse": {
        "common_name": "Computer Mouse",
        "materials_pct": {
            "plastic": 60.0,
            "copper_pcb": 5.0,
            "steel": 8.0,
            "rubber": 10.0,
        },
        "hazardous_substances": {},
        "recovery_priority": "LOW",
        "recovery_method": "Standard WEEE processing",
        "recycling_note": "Low value. Batch with other small peripherals.",
    },
    "hard_drive": {
        "common_name": "Hard Disk Drive (HDD/SSD)",
        "materials_pct": {
            "steel": 35.0,
            "aluminium": 20.0,
            "copper": 5.0,
            "rare_earth_magnets": 2.0,
            "glass_platters": 10.0,
            "plastic": 8.0,
        },
        "hazardous_substances": {},
        "recovery_priority": "MEDIUM",
        "recovery_method": "Shredding → rare earth magnet extraction → metal separation",
        "recycling_note": "Neodymium magnets have high recovery value. Data destruction required.",
    },
    "ram": {
        "common_name": "RAM / Memory Module",
        "materials_pct": {
            "gold": 0.05,
            "silver": 0.2,
            "copper": 20.0,
            "silicon": 15.0,
            "fibreglass": 30.0,
            "tin": 5.0,
        },
        "hazardous_substances": {
            "lead_solder": "Lead-tin solder on older modules",
        },
        "recovery_priority": "HIGH",
        "recovery_method": "Precious metal refining",
        "recycling_note": "Gold content in connectors. Process with PCBs.",
    },
    "cpu": {
        "common_name": "CPU / Processor",
        "materials_pct": {
            "gold": 0.2,
            "silver": 0.1,
            "copper": 15.0,
            "silicon": 20.0,
            "palladium": 0.01,
            "aluminium_heat_spreader": 20.0,
            "ceramic": 10.0,
        },
        "hazardous_substances": {
            "lead_solder": "Solder balls (older CPUs)",
            "beryllium_oxide": "Ceramic substrates in some models - toxic dust",
        },
        "recovery_priority": "HIGH",
        "recovery_method": "Hydrometallurgical gold recovery",
        "recycling_note": "Highest gold concentration of all components. High priority.",
    },
    "capacitor": {
        "common_name": "Capacitor (Electrolytic/Ceramic)",
        "materials_pct": {
            "aluminium": 40.0,
            "paper": 20.0,
            "electrolyte": 15.0,
            "plastic_sleeve": 10.0,
            "tin": 5.0,
        },
        "hazardous_substances": {
            "electrolyte_chemicals": "Corrosive, may contain boric acid compounds",
        },
        "recovery_priority": "MEDIUM",
        "recovery_method": "Aluminium recovery after safe disposal of electrolyte",
        "recycling_note": "Do not puncture. Aluminium recoverable.",
    },
}

# ─── ECONOMIC VALUE ESTIMATOR ─────────────────────────────────────────────────

# USD per kg (approximate market 2024)
MATERIAL_PRICES = {
    "gold": 60000,
    "silver": 800,
    "palladium": 40000,
    "platinum": 30000,
    "copper": 9,
    "aluminium": 2.5,
    "steel": 0.5,
    "iron": 0.4,
    "cobalt": 35,
    "lithium": 20,
    "nickel": 15,
    "tin": 25,
    "rare_earth": 100,
    "rare_earth_magnets": 50,
    "rare_earth_phosphors": 80,
    "silicon": 2,
    "graphite": 5,
    "copper_foil": 9,
    "aluminium_foil": 2.5,
    "copper_pcb": 9,
    "tin_coating": 25,
    "tin_shield": 25,
    "steel_shield": 0.5,
}

COMPONENT_WEIGHTS_G = {
    "circuit_board": 200,
    "mobile_phone": 180,
    "battery": 50,
    "cable": 100,
    "charger": 150,
    "keyboard": 800,
    "monitor": 4000,
    "mouse": 120,
    "hard_drive": 500,
    "ram": 30,
    "cpu": 50,
    "capacitor": 10,
}

# ─── FUNCTIONS ───────────────────────────────────────────────────────────────

def get_material_profile(class_name):
    """Return full material profile for a class."""
    return MATERIAL_PROFILES.get(class_name)


def estimate_value(class_name, quantity=1):
    """Estimate economic recovery value."""
    profile = MATERIAL_PROFILES.get(class_name)
    if not profile:
        return None

    weight_g = COMPONENT_WEIGHTS_G.get(class_name, 100) * quantity
    weight_kg = weight_g / 1000.0
    total = 0.0
    breakdown = {}

    for material, pct in profile["materials_pct"].items():
        material_kg = weight_kg * (pct / 100.0)
        price = MATERIAL_PRICES.get(material, 0)
        value = material_kg * price
        if value >= 0.0001:
            breakdown[material] = round(value, 4)
        total += value

    return {
        "class": class_name,
        "quantity": quantity,
        "weight_g": weight_g,
        "total_value_usd": round(total, 4),
        "value_breakdown": dict(sorted(breakdown.items(), key=lambda x: -x[1])),
        "recovery_priority": profile["recovery_priority"],
        "is_hazardous": len(profile.get("hazardous_substances", {})) > 0,
        "hazardous_substances": list(profile.get("hazardous_substances", {}).keys()),
    }


def classify_batch(detections):
    """Classify a list of detections and generate full report."""
    report = {
        "total_items": len(detections),
        "items": [],
        "summary": {
            "total_value_usd": 0.0,
            "hazardous_count": 0,
            "high_priority_count": 0,
            "routing": {
                "HIGH_VALUE": [],
                "CRITICAL_HAZARD": [],
                "MEDIUM": [],
                "LOW": [],
            }
        }
    }

    for det in detections:
        cls = det.get("class", "unknown")
        conf = det.get("confidence", 0)
        value_info = estimate_value(cls)

        if not value_info:
            continue

        item = {
            "class": cls,
            "confidence": round(conf, 3),
            **value_info
        }
        report["items"].append(item)

        # Update summary
        report["summary"]["total_value_usd"] += value_info["total_value_usd"]
        if value_info["is_hazardous"]:
            report["summary"]["hazardous_count"] += 1
        priority = value_info["recovery_priority"]
        if priority in report["summary"]["routing"]:
            report["summary"]["routing"][priority].append(cls)
        if priority == "HIGH":
            report["summary"]["high_priority_count"] += 1

    report["summary"]["total_value_usd"] = round(
        report["summary"]["total_value_usd"], 4)

    return report


def print_single_class_report(class_name):
    """Print detailed report for a single class."""
    profile = get_material_profile(class_name)
    if not profile:
        print(f"❌ Unknown class: {class_name}")
        print(f"   Available: {', '.join(MATERIAL_PROFILES.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"  📦 {profile['common_name'].upper()}")
    print(f"{'='*60}")
    print(f"  Recovery Priority: {profile['recovery_priority']}")
    print(f"  Recovery Method : {profile['recovery_method']}")
    print(f"  Note: {profile['recycling_note']}")

    print(f"\n  🔬 Material Composition:")
    for mat, pct in sorted(profile["materials_pct"].items(), key=lambda x: -x[1]):
        bar = "█" * int(pct / 3)
        print(f"     {mat:<30} {pct:>5.1f}%  {bar}")

    if profile["hazardous_substances"]:
        print(f"\n  ⚠️  Hazardous Substances:")
        for substance, desc in profile["hazardous_substances"].items():
            print(f"     → {substance}: {desc}")

    value = estimate_value(class_name)
    if value:
        print(f"\n  💰 Economic Value Estimate (1 unit):")
        print(f"     Weight: {value['weight_g']}g")
        print(f"     Total:  ${value['total_value_usd']:.4f}")
        print(f"     Top materials:")
        for mat, val in list(value["value_breakdown"].items())[:5]:
            print(f"       {mat:<25} ${val:.4f}")

    print(f"{'='*60}")


def print_all_classes_summary():
    """Print comparison table of all classes."""
    print(f"\n{'='*75}")
    print(f"  E-WASTE CLASSIFICATION SUMMARY - ALL 12 CLASSES")
    print(f"{'='*75}")
    print(f"  {'Class':<20} {'Priority':<15} {'Value/Unit':>12}  {'Hazardous'}")
    print(f"  {'-'*70}")

    for cls in MATERIAL_PROFILES:
        value = estimate_value(cls)
        profile = MATERIAL_PROFILES[cls]
        hazard_flag = "⚠️ YES" if value["is_hazardous"] else "  no"
        print(f"  {cls:<20} {profile['recovery_priority']:<15} "
              f"${value['total_value_usd']:>10.4f}  {hazard_flag}")

    print(f"{'='*75}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Material Classifier")
    parser.add_argument("--class-name", help="Classify a specific class")
    parser.add_argument("--all-classes", action="store_true",
                        help="Show all class profiles")
    parser.add_argument("--from-json", help="Load detections from JSON file")
    args = parser.parse_args()

    if args.class_name:
        print_single_class_report(args.class_name)

    elif args.all_classes:
        print_all_classes_summary()
        print("\nFor detailed profile: python material_classifier.py --class-name cpu")

    elif args.from_json:
        try:
            with open(args.from_json) as f:
                data = json.load(f)
            detections = data.get("detections", [])
            report = classify_batch(detections)
            print(json.dumps(report, indent=2))
        except FileNotFoundError:
            print(f"❌ File not found: {args.from_json}")
            print("   Run: python predict.py  first")

    else:
        parser.print_help()
        print("\n📌 Examples:")
        print("   python material_classifier.py --all-classes")
        print("   python material_classifier.py --class-name circuit_board")
        print("   python material_classifier.py --from-json predictions.json")


if __name__ == "__main__":
    main()
