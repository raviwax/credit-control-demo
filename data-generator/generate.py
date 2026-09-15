"""Synthetic sales-ledger data for the credit-control demo.

Simulates customers, invoices and cash allocations for a fictional UK B2B supplier,
then computes an independent debtor-ageing cross-check used to QA the DAX measures.

Outputs (written to ../data relative to this file):
    customers.csv, invoices.csv, allocations.csv, expected_ageing.csv

Run:
    python data-generator/generate.py

The seed is fixed, so every run produces identical files.
"""

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED = 42

WINDOW_START = np.datetime64("2024-04-01")
SNAPSHOT = np.datetime64("2026-08-31")  # nothing is dated after this
QA_AS_AT_DATES = [np.datetime64("2026-08-31"), np.datetime64("2026-03-31")]

OUT_DIR = Path(__file__).resolve().parent.parent / "data"

# England and Wales bank holidays; with weekends these are the non-working days
BANK_HOLIDAYS = np.array(
    [
        "2024-04-01", "2024-05-06", "2024-05-27", "2024-08-26", "2024-12-25", "2024-12-26",
        "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-05", "2025-05-26", "2025-08-25",
        "2025-12-25", "2025-12-26",
        "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-04", "2026-05-25", "2026-08-31",
        "2026-12-25", "2026-12-28",
    ],
    dtype="datetime64[D]",
)
CALENDAR = np.busdaycalendar(weekmask="1111100", holidays=BANK_HOLIDAYS)

N_CUSTOMERS = 70
TARGET_INVOICES = 3_000

REGIONS = {
    "North West": 0.12, "North East": 0.06, "Yorkshire": 0.10, "Midlands": 0.14,
    "East of England": 0.09, "London": 0.14, "South East": 0.13, "South West": 0.09,
    "Wales": 0.06, "Scotland": 0.07,
}
INDUSTRIES = {
    "Manufacturing": 0.20, "Distribution": 0.18, "Retail": 0.12, "Construction": 0.16,
    "Food & Drink": 0.14, "Professional Services": 0.10, "Healthcare": 0.10,
}
ACCOUNT_MANAGERS = ["Helen Marsh", "Tom Beckett", "Aisha Rahman", "Gareth Lloyd", "Fiona Kerr"]
PAYMENT_TERMS = {30: 0.70, 45: 0.10, 60: 0.20}

CREDIT_LIMIT_RANGE = (5_000, 250_000)
CREDIT_LIMIT_MEDIAN = 55_000
CREDIT_LIMIT_SIGMA = 0.8
CUSTOMER_SINCE_RANGE = (np.datetime64("2018-01-01"), np.datetime64("2025-12-31"))

# Invented place-style stems keep company names fictional
NAME_HEADS = [
    "Ald", "Brack", "Calder", "Dun", "Elm", "Fern", "Gar", "Hal", "Kel", "Lin", "Mor", "Nether",
    "Pen", "Rook", "Sand", "Thorn", "Whin", "Yar", "Brin", "Cran", "Hes", "Lang", "Sil", "Wold",
]
NAME_TAILS = [
    "combe", "field", "moor", "worth", "leigh", "dale", "ford", "gate", "stead", "wick",
    "brook", "bridge", "hurst", "mere", "thorpe", "holme",
]
NAME_BLOCKLIST = {
    "Thornbridge", "Sandhurst", "Calderdale", "Elmbridge", "Netherfield", "Sandford",
    "Halford", "Garfield", "Cranfield", "Aldgate", "Linford",
}
DESCRIPTORS = {
    "Manufacturing": ["Fabrication", "Engineering", "Precision Components", "Plastics", "Metalworks", "Packaging"],
    "Distribution": ["Logistics", "Distribution", "Wholesale", "Supplies", "Freight Services"],
    "Retail": ["Retail", "Home Stores", "Outfitters", "Garden Centres", "Trading"],
    "Construction": ["Construction", "Building Services", "Civils", "Groundworks", "Contractors"],
    "Food & Drink": ["Foods", "Bakeries", "Fresh Produce", "Dairies", "Catering Supplies"],
    "Professional Services": ["Consulting", "Associates", "Advisory", "Surveyors"],
    "Healthcare": ["Care Homes", "Medical Supplies", "Healthcare", "Dental Group"],
}

# Payment behaviour drives the simulation only and is never exported.
# Lateness is days paid after the due date (negative = early).
PROFILES = {
    "Prompt": {"share": 0.35, "late_mean": -5, "late_sd": 5, "risk": 0},
    "Average": {"share": 0.40, "late_mean": 8, "late_sd": 10, "risk": 1},
    "Slow": {"share": 0.18, "late_mean": 35, "late_sd": 15, "risk": 2},
    "Problem": {"share": 0.07, "late_mean": 70, "late_sd": 25, "risk": 2},
}
# Credit limits follow payment record: riskier profiles are pushed down the limit ranking
LIMIT_TILT = 0.15
# Problem customers pay like Slow ones until a deterioration date, then turn bad
PROBLEM_ONSET_RANGE = (np.datetime64("2025-08-01"), np.datetime64("2026-01-31"))
PROBLEM_NEVER_PAID = 0.25

# Invoicing volume and value
RATE_RANGE = (0.5, 8.0)  # invoices per customer per month, rising with credit limit
SEASONALITY = {3: 1.3, 9: 1.2, 11: 1.2, 8: 0.7, 12: 0.6}
AMOUNT_RANGE = (250, 40_000)
AMOUNT_SIGMA = 0.7
BILLING_TO_LIMIT = 0.26  # typical monthly billing as a share of credit limit
ROUND_HUNDRED_SHARE = 0.15

# Settlement patterns
PARTIAL_SHARE = 0.12  # part-paid first, remainder later
PARTIAL_FIRST_RANGE = (0.30, 0.70)
PARTIAL_GAP_DAYS = (10, 40)
HELD_SHARE = 0.08  # queried/held: paid only after an extra hold, so recent ones are still open
HELD_MEAN_DAYS = 45

# Ageing buckets, identical bounds to the model's Ageing Bucket table
BUCKETS = [
    ("NotYetDue", -100_000, 0),
    ("D1_30", 1, 30),
    ("D31_60", 31, 60),
    ("D61_90", 61, 90),
    ("D91_120", 91, 120),
    ("D120plus", 121, 100_000),
]
MONEY_COLUMNS = ["Outstanding", "Overdue"] + [name for name, _, _ in BUCKETS]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def stream(name: str) -> np.random.Generator:
    """Independent seeded stream per concern, so tuning one parameter doesn't reshuffle the rest."""
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Integrity check failed: {message}")


def working_days(start: np.datetime64, end: np.datetime64) -> np.ndarray:
    days = np.arange(start, end + np.timedelta64(1, "D"), dtype="datetime64[D]")
    return days[np.is_busday(days, busdaycal=CALENDAR)]


def roll_to_working_day(dates: np.ndarray) -> np.ndarray:
    return np.busday_offset(dates, 0, roll="forward", busdaycal=CALENDAR)


def exact_split(shares: dict, n: int) -> np.ndarray:
    """Labels repeated in proportion to shares, using largest remainders so counts sum to n."""
    labels = list(shares)
    raw = np.array([shares[k] for k in labels], dtype=float) * n
    counts = np.floor(raw).astype(int)
    for i in np.argsort(counts - raw, kind="stable")[: n - counts.sum()]:
        counts[i] += 1
    return np.repeat(np.array(labels, dtype=object), counts)


def stratified_labels(rng: np.random.Generator, shares: dict, sizes: np.ndarray, shift: dict) -> np.ndarray:
    """Exact-proportion labels spread evenly down the size ranking (largest first); a label's
    shift moves its whole group further down the ranking."""
    labels = exact_split(shares, len(sizes))
    slots = []
    for key in shares:
        k = int((labels == key).sum())
        offset = rng.random()
        slots += [((i + offset) / k + shift[key], key) for i in range(k)]
    result = np.empty(len(sizes), dtype=object)
    result[np.argsort(-sizes, kind="stable")] = [key for _, key in sorted(slots)]
    return result


def truncated_lognormal(rng: np.random.Generator, median: np.ndarray, sigma: float, lo: float, hi: float) -> np.ndarray:
    """Lognormal draws around per-row medians, redrawing anything outside [lo, hi]."""
    median = np.clip(np.asarray(median, dtype=float), lo, hi)
    out = rng.lognormal(np.log(median), sigma)
    bad = (out < lo) | (out > hi)
    while bad.any():
        out[bad] = rng.lognormal(np.log(median[bad]), sigma)
        bad = (out < lo) | (out > hi)
    return out


def company_names(rng: np.random.Generator, industries: np.ndarray) -> list[str]:
    stems = [
        head + tail
        for head in NAME_HEADS
        for tail in NAME_TAILS
        if head[-1] != tail[0] and head + tail not in NAME_BLOCKLIST
    ]
    chosen = rng.choice(stems, size=len(industries), replace=False)
    names = []
    for stem, industry in zip(chosen, industries):
        descriptor = rng.choice(DESCRIPTORS[str(industry)])
        suffix = "Ltd" if rng.random() < 0.65 else "Limited"
        names.append(f"{stem} {descriptor} {suffix}")
    return names


def gbp(pence: float) -> str:
    return f"GBP {pence / 100:,.2f}"


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------
def make_customers() -> pd.DataFrame:
    n = N_CUSTOMERS
    rng = stream("customers")
    industry = rng.choice(list(INDUSTRIES), size=n, p=list(INDUSTRIES.values()))
    region = rng.choice(list(REGIONS), size=n, p=list(REGIONS.values()))
    since_span = int((CUSTOMER_SINCE_RANGE[1] - CUSTOMER_SINCE_RANGE[0]).astype(int))
    since = CUSTOMER_SINCE_RANGE[0] + rng.integers(0, since_span + 1, n).astype("timedelta64[D]")
    terms = rng.permutation(exact_split(PAYMENT_TERMS, n)).astype(np.int64)
    managers = rng.permutation(np.resize(np.array(ACCOUNT_MANAGERS), n))
    names = company_names(rng, industry)

    credit_limit = truncated_lognormal(
        stream("credit_limits"), np.full(n, CREDIT_LIMIT_MEDIAN), CREDIT_LIMIT_SIGMA, *CREDIT_LIMIT_RANGE
    )
    credit_limit = (np.rint(credit_limit / 500) * 500).astype(np.int64)

    behaviour = stream("behaviour")
    profile = stratified_labels(
        behaviour,
        {k: v["share"] for k, v in PROFILES.items()},
        credit_limit,
        {k: LIMIT_TILT * v["risk"] for k, v in PROFILES.items()},
    )
    onset_span = int((PROBLEM_ONSET_RANGE[1] - PROBLEM_ONSET_RANGE[0]).astype(int))
    onset = PROBLEM_ONSET_RANGE[0] + behaviour.integers(0, onset_span + 1, n).astype("timedelta64[D]")
    onset = np.where(profile == "Problem", onset, np.datetime64("NaT"))

    customers = pd.DataFrame(
        {
            "CustomerName": names,
            "Region": region,
            "Industry": industry,
            "CreditLimit": credit_limit,
            "PaymentTermsDays": terms,
            "AccountManager": managers,
            "CustomerSince": since,
            "Profile": profile,
            "ProblemOnset": onset,
        }
    )
    # Account numbers follow onboarding order
    customers = customers.sort_values("CustomerSince", kind="stable").reset_index(drop=True)
    customers.insert(0, "CustomerID", [f"C{i:03d}" for i in range(1, n + 1)])
    return customers


def calibrate_rates(credit_limit: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """Monthly invoice rate per customer: RATE_RANGE scaled by credit limit, curved so the
    expected invoice count over the window matches TARGET_INVOICES."""
    lo_rate, hi_rate = RATE_RANGE
    position = (credit_limit - CREDIT_LIMIT_RANGE[0]) / (CREDIT_LIMIT_RANGE[1] - CREDIT_LIMIT_RANGE[0])

    def expected(gamma: float) -> float:
        return float(((lo_rate + (hi_rate - lo_rate) * position**gamma) * exposure).sum())

    lo, hi = 0.05, 20.0
    for _ in range(60):
        gamma = np.sqrt(lo * hi)
        if expected(gamma) > TARGET_INVOICES:
            lo = gamma
        else:
            hi = gamma
    return lo_rate + (hi_rate - lo_rate) * position**gamma


def make_invoices(customers: pd.DataFrame) -> pd.DataFrame:
    volume, amounts = stream("invoice_volume"), stream("invoice_amounts")
    days = working_days(WINDOW_START, SNAPSHOT)
    day_month = days.astype("datetime64[M]")
    months = np.unique(day_month)
    since = customers["CustomerSince"].to_numpy(dtype="datetime64[D]")

    # exposure[c, m] = seasonality x share of month m's working days customer c was trading
    exposure = np.zeros((len(customers), len(months)))
    for j, month in enumerate(months):
        month_days = days[day_month == month]
        active = (month_days[None, :] >= since[:, None]).sum(axis=1)
        month_no = int(month.astype(int)) % 12 + 1
        exposure[:, j] = SEASONALITY.get(month_no, 1.0) * active / len(month_days)

    credit_limit = customers["CreditLimit"].to_numpy()
    rate = calibrate_rates(credit_limit, exposure.sum(axis=1))

    counts = volume.poisson(rate[:, None] * exposure)
    cust_idx, inv_dates = [], []
    for c, j in zip(*np.nonzero(counts)):
        month_days = days[day_month == months[j]]
        month_days = month_days[month_days >= since[c]]
        inv_dates.append(volume.choice(month_days, size=counts[c, j]))
        cust_idx.append(np.full(counts[c, j], c))
    cust_idx = np.concatenate(cust_idx)
    inv_dates = np.concatenate(inv_dates)

    # Larger customers raise larger invoices: median sized so monthly billing ~ BILLING_TO_LIMIT x limit
    median = BILLING_TO_LIMIT * credit_limit / rate / np.exp(AMOUNT_SIGMA**2 / 2)
    amount = truncated_lognormal(amounts, median[cust_idx], AMOUNT_SIGMA, *AMOUNT_RANGE)
    round_hundred = amounts.random(len(amount)) < ROUND_HUNDRED_SHARE
    amount = np.clip(np.where(round_hundred, np.round(amount, -2), np.round(amount, 2)), *AMOUNT_RANGE)
    amount_pence = np.rint(amount * 100).astype(np.int64)

    # Invoice numbers run in date order; ties on the same day are shuffled
    order = np.lexsort((volume.random(len(inv_dates)), inv_dates.astype(np.int64)))
    cust_idx, inv_dates, amount_pence = cust_idx[order], inv_dates[order], amount_pence[order]
    terms = customers["PaymentTermsDays"].to_numpy()[cust_idx]

    return pd.DataFrame(
        {
            "InvoiceID": [f"INV-{i:06d}" for i in range(1, len(order) + 1)],
            "CustomerID": customers["CustomerID"].to_numpy()[cust_idx],
            "InvoiceDate": inv_dates,
            "DueDate": inv_dates + terms.astype("timedelta64[D]"),
            "AmountPence": amount_pence,
            "CustomerIndex": cust_idx,
        }
    )


def make_allocations(customers: pd.DataFrame, invoices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rng = stream("settlement")
    n = len(invoices)
    c = invoices["CustomerIndex"].to_numpy()
    inv_date = invoices["InvoiceDate"].to_numpy(dtype="datetime64[D]")
    due = invoices["DueDate"].to_numpy(dtype="datetime64[D]")
    amount = invoices["AmountPence"].to_numpy()
    profile = customers["Profile"].to_numpy()[c]
    onset = customers["ProblemOnset"].to_numpy(dtype="datetime64[D]")[c]

    deteriorated = (profile == "Problem") & (inv_date >= onset)
    behaviour = np.where((profile == "Problem") & ~deteriorated, "Slow", profile)
    late_mean = np.array([PROFILES[b]["late_mean"] for b in behaviour], dtype=float)
    late_sd = np.array([PROFILES[b]["late_sd"] for b in behaviour], dtype=float)
    lateness = np.rint(rng.normal(late_mean, late_sd)).astype(np.int64)

    never_paid = deteriorated & (rng.random(n) < PROBLEM_NEVER_PAID)
    pattern = rng.random(n)
    held = pattern < HELD_SHARE
    partial = (pattern >= HELD_SHARE) & (pattern < HELD_SHARE + PARTIAL_SHARE)
    hold_days = np.where(held, np.rint(rng.exponential(HELD_MEAN_DAYS, n)), 0).astype(np.int64)

    # Receipts land on the next working day and never before the day after invoicing
    first_date = due + (lateness + hold_days).astype("timedelta64[D]")
    first_date = roll_to_working_day(np.maximum(first_date, inv_date + np.timedelta64(1, "D")))
    first_share = rng.uniform(*PARTIAL_FIRST_RANGE, n)
    first_amount = np.where(partial, np.rint(amount * first_share).astype(np.int64), amount)

    gap = rng.integers(PARTIAL_GAP_DAYS[0], PARTIAL_GAP_DAYS[1] + 1, n)
    second_date = roll_to_working_day(first_date + gap.astype("timedelta64[D]"))
    second_amount = amount - first_amount

    # Anything the simulation dates after the snapshot has not happened yet
    first = ~never_paid & (first_date <= SNAPSHOT)
    second = ~never_paid & partial & (second_date <= SNAPSHOT)

    invoice_id = invoices["InvoiceID"].to_numpy()
    customer_id = invoices["CustomerID"].to_numpy()
    allocations = pd.DataFrame(
        {
            "InvoiceID": np.concatenate([invoice_id[first], invoice_id[second]]),
            "CustomerID": np.concatenate([customer_id[first], customer_id[second]]),
            "AllocationDate": np.concatenate([first_date[first], second_date[second]]),
            "AmountPence": np.concatenate([first_amount[first], second_amount[second]]),
        }
    )
    allocations = allocations.sort_values(["AllocationDate", "InvoiceID"], kind="stable").reset_index(drop=True)
    allocations.insert(0, "AllocationID", [f"ALC-{i:06d}" for i in range(1, len(allocations) + 1)])

    stats = {
        "held": int(held.sum()),
        "partial": int(partial.sum()),
        "deteriorated": int(deteriorated.sum()),
        "never_paid": int(never_paid.sum()),
    }
    return allocations, stats


# -----------------------------------------------------------------------------
# QA: ageing computed independently of the model
# -----------------------------------------------------------------------------
def ageing(customers: pd.DataFrame, invoices: pd.DataFrame, allocations: pd.DataFrame, as_at: np.datetime64) -> pd.DataFrame:
    as_at = pd.Timestamp(as_at)
    inv = invoices.loc[invoices["InvoiceDate"] <= as_at, ["InvoiceID", "CustomerID", "DueDate", "AmountPence"]]
    paid = allocations.loc[allocations["AllocationDate"] <= as_at].groupby("InvoiceID")["AmountPence"].sum()
    balance = inv["AmountPence"] - inv["InvoiceID"].map(paid).fillna(0).astype(np.int64)

    # Balances are whole pence, so "> GBP 0.005" is simply "> 0"
    items = inv.assign(Balance=balance).loc[lambda t: t["Balance"] > 0].copy()
    days = (as_at - items["DueDate"]).dt.days
    items["Outstanding"] = items["Balance"]
    items["Overdue"] = items["Balance"].where(days >= 1, 0)
    for name, lo, hi in BUCKETS:
        items[name] = items["Balance"].where(days.between(lo, hi), 0)
    items["OverdueDays"] = days.where(days >= 1)

    grouped = items.groupby("CustomerID").agg(
        **{col: (col, "sum") for col in MONEY_COLUMNS},
        OldestOverdueDays=("OverdueDays", "max"),
        OpenInvoices=("InvoiceID", "count"),
    )
    result = customers[["CustomerID", "CustomerName"]].merge(grouped, left_on="CustomerID", right_index=True, how="left")
    result[MONEY_COLUMNS + ["OpenInvoices"]] = result[MONEY_COLUMNS + ["OpenInvoices"]].fillna(0).astype(np.int64)

    total = {
        "CustomerID": "TOTAL",
        "CustomerName": "TOTAL",
        **{col: result[col].sum() for col in MONEY_COLUMNS},
        "OldestOverdueDays": result["OldestOverdueDays"].max(),
        "OpenInvoices": result["OpenInvoices"].sum(),
    }
    result = pd.concat([result, pd.DataFrame([total])], ignore_index=True)
    result["OldestOverdueDays"] = result["OldestOverdueDays"].astype("Int64")
    result.insert(0, "AsAtDate", as_at)
    return result


def check_integrity(customers: pd.DataFrame, invoices: pd.DataFrame, allocations: pd.DataFrame) -> None:
    inv_days = invoices["InvoiceDate"].to_numpy(dtype="datetime64[D]")
    alc_days = allocations["AllocationDate"].to_numpy(dtype="datetime64[D]")
    require(np.is_busday(inv_days, busdaycal=CALENDAR).all(), "invoice dated on a non-working day")
    require(np.is_busday(alc_days, busdaycal=CALENDAR).all(), "allocation dated on a non-working day")
    require(((inv_days >= WINDOW_START) & (inv_days <= SNAPSHOT)).all(), "invoice outside the activity window")
    require((alc_days <= SNAPSHOT).all(), "allocation after the snapshot")
    require((allocations["AmountPence"] > 0).all(), "non-positive allocation")
    require(invoices["InvoiceID"].is_unique and allocations["AllocationID"].is_unique, "duplicate IDs")
    require(invoices["CustomerID"].isin(customers["CustomerID"]).all(), "invoice with unknown customer")

    merged = allocations.merge(
        invoices[["InvoiceID", "CustomerID", "InvoiceDate", "AmountPence"]],
        on="InvoiceID", how="left", suffixes=("", "Invoice"), validate="many_to_one",
    )
    require(merged["InvoiceDate"].notna().all(), "allocation with unknown invoice")
    require((merged["CustomerID"] == merged["CustomerIDInvoice"]).all(), "allocation customer differs from invoice")
    require((merged["AllocationDate"] > merged["InvoiceDate"]).all(), "allocation on or before invoice date")
    allocated = merged.groupby("InvoiceID").agg(Allocated=("AmountPence", "sum"), Amount=("AmountPenceInvoice", "first"))
    require((allocated["Allocated"] <= allocated["Amount"]).all(), "invoice over-allocated")


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def write_csv(frame: pd.DataFrame, name: str, pence_columns: list[str]) -> None:
    out = frame.copy()
    for col in pence_columns:
        out[col] = out[col] / 100
    out.to_csv(OUT_DIR / name, index=False, date_format="%Y-%m-%d", float_format="%.2f", lineterminator="\n")


def report(customers, invoices, allocations, expected, stats) -> bool:
    snap = expected[expected["AsAtDate"] == pd.Timestamp(SNAPSHOT)]
    total = snap[snap["CustomerID"] == "TOTAL"].iloc[0]
    per_customer = snap[snap["CustomerID"] != "TOTAL"].merge(
        customers[["CustomerID", "CreditLimit", "Profile"]], on="CustomerID"
    )
    per_customer["Utilisation"] = per_customer["Outstanding"] / (per_customer["CreditLimit"] * 100)
    over_limit = per_customer[per_customer["Utilisation"] > 1]
    top10 = per_customer.nlargest(10, "Overdue")

    outstanding = total["Outstanding"]
    overdue_pct = total["Overdue"] / outstanding
    over120_pct = total["D120plus"] / outstanding
    snap_items = ageing_items_over_120(invoices, allocations, SNAPSHOT)

    print("Rows")
    print(f"  customers.csv        {len(customers):>6,}")
    print(f"  invoices.csv         {len(invoices):>6,}")
    print(f"  allocations.csv      {len(allocations):>6,}")
    print(f"  expected_ageing.csv  {len(expected):>6,}")
    print(
        f"  (settlement: {stats['held']} held, {stats['partial']} part-paid, "
        f"{stats['deteriorated']} after Problem onset, of which {stats['never_paid']} never paid)"
    )

    print(f"\nLedger as at {pd.Timestamp(SNAPSHOT):%d %b %Y}")
    print(f"  {'Outstanding':<15}{gbp(outstanding):>20}")
    print(f"  {'Overdue':<15}{gbp(total['Overdue']):>20}  {overdue_pct:6.1%}")
    labels = ["Not yet due", "1-30 days", "31-60 days", "61-90 days", "91-120 days", "Over 120 days"]
    for label, (col, _, _) in zip(labels, BUCKETS):
        print(f"  {label:<15}{gbp(total[col]):>20}  {total[col] / outstanding:6.1%}")
    print(f"  Open invoices  {total['OpenInvoices']:>20,}  ({total['OpenInvoices'] / len(invoices):.1%} of all invoices)")
    print(f"  120+ days      {snap_items[0]:>20,} invoices across {snap_items[1]} customers")

    mar = expected[(expected["AsAtDate"] == pd.Timestamp("2026-03-31")) & (expected["CustomerID"] == "TOTAL")].iloc[0]
    print(f"\nLedger as at 31 Mar 2026: outstanding {gbp(mar['Outstanding'])}, overdue {mar['Overdue'] / mar['Outstanding']:.1%}")

    print("\nTop 10 overdue customers (profile is console-only, never exported)")
    print(f"  {'ID':<5}{'Customer':<46}{'Profile':<9}{'Overdue':>14}{'Outstanding':>14}{'Limit':>10}{'Util':>7}{'Oldest':>7}")
    for row in top10.itertuples():
        print(
            f"  {row.CustomerID:<5}{row.CustomerName:<46}{row.Profile:<9}{row.Overdue / 100:>14,.2f}"
            f"{row.Outstanding / 100:>14,.2f}{row.CreditLimit:>10,}{row.Utilisation:>7.0%}{str(row.OldestOverdueDays):>7}"
        )
    print("\nOver credit limit")
    for row in over_limit.sort_values("Utilisation", ascending=False).itertuples():
        print(f"  {row.CustomerID:<5}{row.CustomerName:<46}{row.Profile:<9}{row.Utilisation:>7.0%}")

    problem_in_top10 = int((top10["Profile"] == "Problem").sum())
    checks = [
        ("Invoices ~3,000 (2,700-3,300)", 2_700 <= len(invoices) <= 3_300, f"{len(invoices):,}"),
        ("Outstanding GBP 1.5m-2.5m", 150_000_000 <= outstanding <= 250_000_000, gbp(outstanding)),
        ("Overdue 30-40% of outstanding", 0.30 <= overdue_pct <= 0.40, f"{overdue_pct:.1%}"),
        ("Every ageing bucket non-empty", all(total[col] > 0 for col, _, _ in BUCKETS), ""),
        ("Over 120 days 3-8% of outstanding", 0.03 <= over120_pct <= 0.08, f"{over120_pct:.1%}"),
        ("3-6 customers over credit limit", 3 <= len(over_limit) <= 6, str(len(over_limit))),
        ("2+ Problem customers in top 10 overdue", problem_in_top10 >= 2, str(problem_in_top10)),
    ]
    print("\nSanity targets")
    for label, ok, value in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<42}{value}")
    return all(ok for _, ok, _ in checks)


def ageing_items_over_120(invoices: pd.DataFrame, allocations: pd.DataFrame, as_at: np.datetime64) -> tuple[int, int]:
    as_at = pd.Timestamp(as_at)
    paid = allocations.loc[allocations["AllocationDate"] <= as_at].groupby("InvoiceID")["AmountPence"].sum()
    inv = invoices[invoices["InvoiceDate"] <= as_at]
    balance = inv["AmountPence"] - inv["InvoiceID"].map(paid).fillna(0)
    old = inv[(balance > 0) & ((as_at - inv["DueDate"]).dt.days > 120)]
    return len(old), old["CustomerID"].nunique()


def main() -> int:
    customers = make_customers()
    invoices = make_invoices(customers)
    allocations, stats = make_allocations(customers, invoices)
    check_integrity(customers, invoices, allocations)
    expected = pd.concat([ageing(customers, invoices, allocations, d) for d in QA_AS_AT_DATES], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    customer_columns = ["CustomerID", "CustomerName", "Region", "Industry", "CreditLimit",
                        "PaymentTermsDays", "AccountManager", "CustomerSince"]
    write_csv(customers[customer_columns], "customers.csv", [])
    write_csv(invoices.rename(columns={"AmountPence": "Amount"}).drop(columns="CustomerIndex"), "invoices.csv", ["Amount"])
    write_csv(allocations.rename(columns={"AmountPence": "Amount"}), "allocations.csv", ["Amount"])
    write_csv(expected, "expected_ageing.csv", MONEY_COLUMNS)

    ok = report(customers, invoices, allocations, expected, stats)
    print(f"\nWrote 4 files to {OUT_DIR.name}/")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
