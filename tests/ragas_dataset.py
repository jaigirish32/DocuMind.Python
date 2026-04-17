# tests/ragas_dataset.py

SKID_ASSEMBLY_TESTS = [
    {
        "question":     "What is the drawing number?",
        "ground_truth": "P08-03906-041",
    },
    {
        "question":     "What is the revision number?",
        "ground_truth": "B",
    },
    {
        "question":     "What is the part number for the mass flow controller?",
        "ground_truth": "80458-001",
    },
    {
        "question":     "What is the motor specification?",
        "ground_truth": "1HP, 208-230/460 VAC, 3 Phase, 60Hz, 1150 RPM",
    },
    {
        "question":     "What is the total number of sheets?",
        "ground_truth": "14 sheets",
    },
]

CABLE_TESTS = [
    {
        "question":     "What is the maximum voltage rating of the cable?",
        "ground_truth": "300V",
    },
    {
        "question":     "What is the conductor resistance at 20 degrees C?",
        "ground_truth": "40.10 ohms per kilometer maximum",
    },
    {
        "question":     "What is the nominal sheath thickness?",
        "ground_truth": "1.20 mm",
    },
    {
        "question":     "What is the conductor material?",
        "ground_truth": "Annealed Tinned Copper",
    },
    {
        "question":     "What is the diameter over sheath?",
        "ground_truth": "13.30 mm",
    },
]

POWER_SUPPLY_TESTS = [
    {
        "question":     "What is the specification number?",
        "ground_truth": "80283",
    },
    {
        "question":     "What is the output power with convection cooling?",
        "ground_truth": "30W convection, 40W with 15 cfm airflow",
    },
    {
        "question":     "What is the MTBF of the power supply?",
        "ground_truth": "greater than 500000 hours",
    },
    {
        "question":     "What are the physical dimensions of the power supply?",
        "ground_truth": "3.60 x 2.44 x 1.00 inches or 91.5 x 62.0 x 25.4 mm",
    },
    {
        "question":     "What is the input voltage range?",
        "ground_truth": "90 to 264 VAC",
    },
]

# Map document IDs to test cases
DOCUMENT_TEST_MAP = {
    "P08-03906-041B_ASSEMBLY__SKID__AM300PIGMF__RE-0043__AS-BUILT__ECN_REQ_FOR_PARTS": SKID_ASSEMBLY_TESTS,
    "12_core_cable_datasheet":  CABLE_TESTS,
    "PowerSupply_Datasheet":    POWER_SUPPLY_TESTS,
}