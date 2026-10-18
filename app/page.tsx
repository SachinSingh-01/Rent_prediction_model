"use client";

import { FormEvent, useState } from "react";

type FormValues = {
  city: string;
  locality: string;
  bhk: string;
  size: string;
  bathrooms: string;
  floor: string;
  totalFloors: string;
  areaType: string;
  furnishingStatus: string;
  tenantPreferred: string;
  pointOfContact: string;
};

const initialValues: FormValues = {
  city: "",
  locality: "",
  bhk: "",
  size: "",
  bathrooms: "",
  floor: "",
  totalFloors: "",
  areaType: "",
  furnishingStatus: "",
  tenantPreferred: "",
  pointOfContact: "",
};

const cities = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"];

type FieldProps = {
  label: string;
  name: keyof FormValues;
  value: string;
  onChange: (name: keyof FormValues, value: string) => void;
  type?: "text" | "number";
  placeholder?: string;
  min?: number;
  max?: number;
  step?: number;
  suffix?: string;
  options?: string[];
  hint?: string;
  required?: boolean;
};

function Field({
  label,
  name,
  value,
  onChange,
  type = "text",
  placeholder,
  min,
  max,
  step,
  suffix,
  options,
  hint,
  required = true,
}: FieldProps) {
  const shared = {
    id: name,
    name,
    value,
    required,
    onChange: (event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
      onChange(name, event.target.value),
    className: `field-control${suffix ? " field-with-suffix" : ""}`,
  };

  return (
    <div className="field">
      <label htmlFor={name} className="field-label">
        {label}
      </label>
      <div className="field-input-wrap">
        {options ? (
          <select {...shared}>
            <option value="" disabled>
              {placeholder ?? "Select an option"}
            </option>
            {options.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        ) : (
          <input
            {...shared}
            type={type}
            placeholder={placeholder}
            min={min}
            max={max}
            step={step}
          />
        )}
        {suffix && <span className="field-suffix">{suffix}</span>}
      </div>
      {hint && <span className="field-hint">{hint}</span>}
    </div>
  );
}

function ArrowIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true" className="icon">
      <path d="M4 10h11m-4.5-4.5L15 10l-4.5 4.5" />
    </svg>
  );
}

function PinIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true" className="pin-icon">
      <path d="M16 8.3c0 4.4-6 9.2-6 9.2S4 12.7 4 8.3a6 6 0 1 1 12 0Z" />
      <circle cx="10" cy="8" r="2" />
    </svg>
  );
}

export default function Home() {
  const [values, setValues] = useState<FormValues>(initialValues);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [rent, setRent] = useState<number | null>(null);
  const apiUrl = process.env.NEXT_PUBLIC_PREDICTION_API_URL?.trim();

  function updateValue(name: keyof FormValues, value: string) {
    setValues((current) => ({ ...current, [name]: value }));
    setError("");
    setRent(null);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setRent(null);

    if (Number(values.floor) > Number(values.totalFloors)) {
      setError("Floor number cannot be greater than the total floors.");
      return;
    }

    if (!apiUrl) {
      setError("Prediction is unavailable because the model API has not been configured yet.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${apiUrl.replace(/\/$/, "")}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city: values.city,
          locality: values.locality,
          bhk: Number(values.bhk),
          size: Number(values.size),
          bathrooms: Number(values.bathrooms),
          floor: Number(values.floor),
          total_floors: Number(values.totalFloors),
          area_type: values.areaType,
          furnishing_status: values.furnishingStatus,
          tenant_preferred: values.tenantPreferred,
          point_of_contact: values.pointOfContact,
        }),
      });
      if (!response.ok) throw new Error(`The model API returned an error (${response.status}).`);
      const result: unknown = await response.json();
      const predictedRent =
        typeof result === "object" && result !== null && "predicted_rent" in result
          ? result.predicted_rent
          : typeof result === "object" && result !== null && "rent" in result
            ? result.rent
            : null;
      if (typeof predictedRent !== "number" || !Number.isFinite(predictedRent) || predictedRent < 0) {
        throw new Error("The model API response must include a numeric predicted_rent value.");
      }
      setRent(predictedRent);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "We could not reach the prediction service. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  const formatter = new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  });

  return (
    <main>
      <header className="site-header">
        <a href="#home" className="brand" aria-label="Rentwise home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 28 28">
              <path d="m4 13 10-8 10 8v10H4V13Z" />
              <path d="M11 23v-7h6v7M2 13l12-10 12 10" />
            </svg>
          </span>
          <span>rentwise<span className="brand-period">.</span></span>
        </a>
        <nav className="site-nav" aria-label="Main navigation">
          <a href="#predict">Predict rent</a>
          <a href="#about">About the project</a>
        </nav>
        <a className="header-cta" href="#predict">
          Get an estimate <ArrowIcon />
        </a>
      </header>

      <section className="hero" id="home">
        <div className="hero-copy">
          <div className="eyebrow"><span className="eyebrow-dot" /> A machine learning project</div>
          <h1>A place to call home.<br /><em>A rent to plan for.</em></h1>
          <p className="hero-description">
            Explore how thoughtful data and machine learning can help make sense of the rental market across India.
          </p>
          <a className="primary-button" href="#predict">Explore your estimate <ArrowIcon /></a>
          <div className="hero-caption"><span className="caption-line" />Built around real rental listing data</div>
        </div>

        <div className="hero-art" aria-label="Illustration of a contemporary home">
          <div className="art-sun" />
          <div className="art-label"><PinIcon /><span>Finding your fit<br /><strong>starts with the details</strong></span></div>
          <div className="art-building">
            <div className="building-roof" />
            <div className="building-body">
              <div className="building-window window-one"><i /><i /></div>
              <div className="building-window window-two"><i /><i /></div>
              <div className="building-door" />
              <div className="building-base" />
            </div>
          </div>
          <div className="art-foreground"><span /><span /><span /><span /><span /></div>
          <div className="art-stamp"><span>HOME</span><span>BEGINS HERE</span><svg viewBox="0 0 44 18" aria-hidden="true"><path d="M2 9h37m-6-6 7 6-7 6" /></svg></div>
        </div>
      </section>

      <section className="form-section" id="predict">
        <div className="section-heading">
          <div>
            <div className="eyebrow section-eyebrow"><span className="eyebrow-dot" /> Your next step</div>
            <h2>Let’s find your <em>number.</em></h2>
            <p>Share a few details about the home you have in mind.</p>
          </div>
          <div className="step-indicator"><span>01</span><i /> PROPERTY DETAILS</div>
        </div>

        <div className="form-card">
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <Field label="City" name="city" value={values.city} onChange={updateValue} options={cities} placeholder="Choose a city" />
              <Field label="Locality" name="locality" value={values.locality} onChange={updateValue} placeholder="e.g. Indiranagar" />
              <Field label="BHK" name="bhk" value={values.bhk} onChange={updateValue} type="number" min={1} max={20} placeholder="e.g. 2" />
              <Field label="Property size" name="size" value={values.size} onChange={updateValue} type="number" min={1} step={1} placeholder="e.g. 1,200" suffix="sq ft" />
              <Field label="Bathrooms" name="bathrooms" value={values.bathrooms} onChange={updateValue} type="number" min={1} max={20} placeholder="e.g. 2" />
              <Field label="Floor number" name="floor" value={values.floor} onChange={updateValue} type="number" min={0} placeholder="e.g. 3" hint="Use 0 for ground floor" />
              <Field label="Total floors" name="totalFloors" value={values.totalFloors} onChange={updateValue} type="number" min={1} placeholder="e.g. 8" />
              <Field label="Area type" name="areaType" value={values.areaType} onChange={updateValue} options={["Super Area", "Carpet Area", "Built Area"]} />
              <Field label="Furnishing status" name="furnishingStatus" value={values.furnishingStatus} onChange={updateValue} options={["Furnished", "Semi-Furnished", "Unfurnished"]} />
              <Field label="Tenant preference" name="tenantPreferred" value={values.tenantPreferred} onChange={updateValue} options={["Bachelors", "Family", "Bachelors/Family"]} />
              <Field label="Point of contact" name="pointOfContact" value={values.pointOfContact} onChange={updateValue} options={["Contact Owner", "Contact Agent", "Contact Builder"]} />
            </div>

            <div className="form-footer">
              <p><span className="lock-icon">✳</span> Your details help shape a more relevant estimate.</p>
              <button className="primary-button submit-button" type="submit" disabled={loading}>
                {loading ? <><span className="spinner" /> Calculating…</> : <>Estimate monthly rent <ArrowIcon /></>}
              </button>
            </div>
          </form>

          <div className={`result-panel${rent !== null ? " result-success" : error ? " result-error" : ""}`} aria-live="polite" role="status">
            <div className="result-icon"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 19V8m5 11V5m5 14v-7m5 7V9M2 19h20" /></svg></div>
            <div className="result-copy">
              <span className="result-label">{rent !== null ? "YOUR ESTIMATED MONTHLY RENT" : error ? "PREDICTION UNAVAILABLE" : "YOUR ESTIMATE WILL APPEAR HERE"}</span>
              <strong className="result-value">{rent !== null ? formatter.format(rent) : error ? "Let’s connect the model" : "₹ —"}</strong>
              <span className="result-message">
                {rent !== null
                  ? "Generated by the configured prediction model."
                  : error
                    ? error
                    : "A real model API is required to generate a prediction. No sample value is shown."}
              </span>
            </div>
            <div className="result-badge"><span className={`status-dot${apiUrl ? " status-ready" : ""}`} />{apiUrl ? "API CONNECTED" : "API NOT CONNECTED"}</div>
          </div>
          {!apiUrl && <p className="integration-note">Integration placeholder · Set <code>NEXT_PUBLIC_PREDICTION_API_URL</code> to enable real predictions.</p>}
        </div>
      </section>

      <section className="about-section" id="about">
        <div className="about-number">01 <span>— OUR APPROACH</span></div>
        <div className="about-copy">
          <h2>Good decisions begin<br />with <em>good information.</em></h2>
          <p>Rentwise is the front end for a house rent prediction machine learning project. The experience is designed around rental listing data and the practical details people use when comparing a home.</p>
          <p>The estimate will come from the project’s trained model once it is served through an API. Until that connection is in place, this page is an honest preview of the experience—not a source of made-up rent figures.</p>
        </div>
        <div className="about-aside"><span className="aside-mark">R.</span><span>Made for a more<br />considered move.</span></div>
      </section>

      <footer className="site-footer"><a href="#home" className="brand footer-brand"><span className="brand-mark" aria-hidden="true"><svg viewBox="0 0 28 28"><path d="m4 13 10-8 10 8v10H4V13Z" /><path d="M11 23v-7h6v7M2 13l12-10 12 10" /></svg></span><span>rentwise<span className="brand-period">.</span></span></a><span>A house rent prediction project</span><a href="#predict">Make a fresh estimate <ArrowIcon /></a></footer>
    </main>
  );
}
