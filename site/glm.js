// GLM scoring engine for the underwriting calculator.
// AI-assisted (Databricks Genie Code), issue #182.
//
// Scores one policy with Gerard's Week 2 model C0, using site/data/model.json
// (built by site/scripts/build_model.py). Runs in the browser (Quarto OJS)
// and in Node (site/scripts/check_engine.py tests it against Python).
//
//   cost of a component = exp(intercept + sum of term contributions
//                             + trend to 2025 + ln(Pareto adjustment)) x exposure
//   as is   = small + large
//   final   = as is, then each level step for the segment, in order
//
// A missing value (null / undefined / "") scores the term's MISSING level.

const isMissing = (x) => x === null || x === undefined || x === "" || (typeof x === "number" && Number.isNaN(x));

function inBounds(x, lo, loInc, hi, hiInc) {
  const okLo = lo === null || (loInc ? x >= lo : x > lo);
  const okHi = hi === null || (hiInc ? x <= hi : x < hi);
  return okLo && okHi;
}

export function splineValue(term, x) {
  for (const p of term.pieces) {
    if (inBounds(x, p.lower, p.lower_included, p.upper, p.upper_included)) {
      const d = x - p.origin;
      const [a0, a1, a2, a3] = p.a;
      return a0 + a1 * d + a2 * d * d + a3 * d * d * d;
    }
  }
  throw new Error(`no spline piece for ${term.term} at ${x}`);
}

// Log contribution of one term, with the level it scored as.
export function termContribution(term, policy) {
  const x = policy[term.input];
  switch (term.kind) {
    case "categorical": {
      const key = isMissing(x) ? "MISSING" : String(x);
      if (key in term.levels) return { level: key, contribution: term.levels[key] };
      return { level: `${key} (scores as base)`, contribution: 0 };
    }
    case "band": {
      if (isMissing(x)) return { level: "MISSING", contribution: term.missing ?? 0 };
      const v = Number(x);
      for (const b of term.bands) {
        if (inBounds(v, b.lower, false, b.upper, true)) return { level: b.level, contribution: b.contribution };
      }
      throw new Error(`no band for ${term.term} at ${v}`);
    }
    case "spline": {
      if (isMissing(x)) return { level: "MISSING", contribution: term.missing };
      return { level: String(x), contribution: splineValue(term, Number(x)) };
    }
    case "log_si": {
      if (isMissing(x)) throw new Error("sum insured is required");
      let si = Number(x);
      if (si >= 1e6) si = si / 1000; // data fix D-02 (contents)
      return { level: String(x), contribution: term.coefficient * Math.log(Math.max(si, 1)) };
    }
    default:
      throw new Error(`unknown term kind ${term.kind}`);
  }
}

export function scoreComponent(comp, policy) {
  const exposure = isMissing(policy.exposure) ? 1 : Number(policy.exposure);
  const terms = comp.terms.map((t) => {
    const r = termContribution(t, policy);
    return { term: t.term, input: t.input, kind: t.kind, ...r, relativity: Math.exp(r.contribution) };
  });
  const sumTerms = terms.reduce((s, t) => s + t.contribution, 0);
  const eta = comp.intercept + sumTerms + comp.trend_log_2025 + comp.scale_log + Math.log(exposure);
  return {
    intercept: comp.intercept,
    base: Math.exp(comp.intercept), // cost per policy-year with every factor at its base level
    terms,
    trend_multiplier: Math.exp(comp.trend_log_2025),
    scale: Math.exp(comp.scale_log),
    exposure,
    eta,
    cost: Math.exp(eta),
  };
}

export function scorePolicy(model, line, policy) {
  const L = model.lines[line];
  if (!L) throw new Error(`unknown line ${line}`);
  const components = {};
  let asIs = 0;
  for (const [name, comp] of Object.entries(L.components)) {
    components[name] = scoreComponent(comp, policy);
    asIs += components[name].cost;
  }
  const segment = policy.new_business === true || policy.new_business === "true" ? "nb" : "ren";
  let final = asIs;
  const steps = [];
  for (const s of L.segments[segment]) {
    final = s.op === "times" ? final * s.value : final / s.value;
    steps.push({ ...s, running: final });
  }
  return { line, segment, components, as_is: asIs, steps, final };
}

// Office premium from the annual risk premium, mirroring Gerard's
// python/office_premium.py build_up() (audit A31), per policy-year:
//   office = (risk x (1 + cat reinsurance, property lines only) + fixed expense)
//            / (1 - variable expense - commission for the channel - premium tax - profit margin)
// `rates` maps expense item -> value for the market (emerging); a missing
// comparison-site commission must already be filled (build_reference.py does it).
export const PROPERTY_LINES = ["buildings", "contents"];
export function officePremium(riskAnnual, line, channel, rates) {
  const commission = rates[`commission_${channel}`];
  if (commission === undefined || commission === null) throw new Error(`no commission rate for channel ${channel}`);
  const cat = PROPERTY_LINES.includes(line) ? rates.cat_reinsurance * riskAnnual : 0;
  const loading = rates.variable_expense + commission + rates.premium_tax + rates.target_profit_margin;
  if (loading >= 1) throw new Error("premium-proportional loadings reach 100% of premium");
  const fixed = rates.fixed_expense_per_policy;
  const office = (riskAnnual + cat + fixed) / (1 - loading);
  return {
    risk_premium: riskAnnual, cat_reinsurance: cat, fixed_expense: fixed,
    variable_expense: office * rates.variable_expense, commission: office * commission,
    premium_tax: office * rates.premium_tax, profit_margin: office * rates.target_profit_margin,
    commission_rate: commission, loading, office_premium: office,
  };
}

// Default policy for a line: every select at its base level, numbers at the
// middle of their range.
export function defaultPolicy(model, line) {
  const p = { exposure: 1, new_business: false };
  for (const [k, inp] of Object.entries(model.lines[line].inputs)) {
    p[k] = inp.type === "select" ? inp.default : Math.round(((inp.min + inp.max) / 2) / inp.step) * inp.step;
  }
  return p;
}
