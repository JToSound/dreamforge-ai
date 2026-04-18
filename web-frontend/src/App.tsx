import { useEffect, useMemo, useState, type CSSProperties, type MouseEvent } from "react";
import "./App.css";

type ThemeMode = "dark" | "light";

interface ServiceItem {
  title: string;
  description: string;
  tag: string;
}

interface FeatureItem {
  title: string;
  body: string;
  metric: string;
}

interface ShowcaseItem {
  title: string;
  label: string;
}

interface CaseStudyItem {
  client: string;
  challenge: string;
  outcome: string;
}

interface FaqItem {
  question: string;
  answer: string;
}

interface DirectionItem {
  id: string;
  name: string;
  color: string;
  type: string;
  motion: string;
  brandFit: string;
}

const navItems = [
  { label: "About", href: "#about" },
  { label: "Services", href: "#services" },
  { label: "Features", href: "#features" },
  { label: "Showcase", href: "#showcase" },
  { label: "Case Studies", href: "#cases" },
  { label: "FAQ", href: "#faq" },
];

const services: ServiceItem[] = [
  {
    title: "Brand Experience Direction",
    description: "Translate brand essence into a cinematic, memorable digital language.",
    tag: "Strategy",
  },
  {
    title: "Liquid Interface Design",
    description: "Build premium UI layers with metal reflections and refined glass depth.",
    tag: "UI Systems",
  },
  {
    title: "Frontend Motion Engineering",
    description: "Craft interaction pacing, micro-feedback, and high-end section choreography.",
    tag: "Motion",
  },
  {
    title: "Narrative Information Architecture",
    description: "Sequence content for clarity, persuasion, and high-value conversion.",
    tag: "UX",
  },
  {
    title: "Creative Technology Integration",
    description: "Integrate WebGL and advanced visuals without sacrificing performance.",
    tag: "Creative Dev",
  },
  {
    title: "Design System Delivery",
    description: "Ship scalable tokens and component specs for long-term maintainability.",
    tag: "Handoff",
  },
];

const features: FeatureItem[] = [
  {
    title: "Floating Island Layout",
    body: "Spatial modules orbit around key narratives so every screen feels intentional and immersive.",
    metric: "12-column desktop rhythm",
  },
  {
    title: "Liquid Metal Core Visual",
    body: "Hero identity object anchors attention with controlled reflection and depth cues.",
    metric: "Single focal effect per screen",
  },
  {
    title: "Micro-interaction Grammar",
    body: "Buttons, cards, and inputs share a tactile motion language with restrained premium feedback.",
    metric: "140–320ms interaction cadence",
  },
  {
    title: "Performance-safe Motion Stack",
    body: "Animations prioritize transform and opacity with automatic reduced-motion fallbacks.",
    metric: "Adaptive by capability tier",
  },
];

const showcaseItems: ShowcaseItem[] = [
  { title: "Material Surface Study", label: "Liquid Metal" },
  { title: "Immersive Product Narrative", label: "Spatial Story" },
  { title: "High-contrast Editorial Grid", label: "Typography" },
  { title: "Adaptive Motion Language", label: "Interaction" },
  { title: "Premium Commerce Journey", label: "Conversion" },
  { title: "Cross-device Fidelity", label: "Responsive" },
];

const caseStudies: CaseStudyItem[] = [
  {
    client: "Aether Labs",
    challenge: "Complex AI product looked technical but emotionally flat.",
    outcome: "Reframed into immersive product storytelling and lifted qualified leads by 37%.",
  },
  {
    client: "Obsidian Capital",
    challenge: "Enterprise trust perception lagged behind service quality.",
    outcome: "Introduced premium spatial UI with stronger trust signals and 2.1x demo request growth.",
  },
  {
    client: "Nocturne Audio",
    challenge: "Brand lacked a distinct digital identity in a crowded segment.",
    outcome: "Built a future-classic visual system that doubled branded search within one quarter.",
  },
];

const processSteps = [
  {
    title: "Discover",
    body: "Brand, audience, and positioning alignment with measurable business intent.",
  },
  {
    title: "Design",
    body: "High-fidelity concept, visual system, and responsive behavior across breakpoints.",
  },
  {
    title: "Build",
    body: "Componentized frontend implementation with motion tiers and quality constraints.",
  },
  {
    title: "Optimize",
    body: "Iterate narrative, performance, and conversion pathways post-launch.",
  },
];

const faqs: FaqItem[] = [
  {
    question: "Is this style usable for conversion-focused websites?",
    answer:
      "Yes. The visual language is premium, but information hierarchy stays explicit, with clear CTA pathways and strong readability.",
  },
  {
    question: "Will this run smoothly on mobile devices?",
    answer:
      "Yes. Heavy visual layers are progressively enhanced. Mobile receives a simplified composition while keeping the brand signature.",
  },
  {
    question: "How do you avoid the generic AI aesthetic?",
    answer:
      "By pairing strict editorial structure with restrained material effects, avoiding overused neon gradients and template patterns.",
  },
  {
    question: "Can this scale into a larger product site?",
    answer:
      "Yes. Tokens, components, and interaction rules are built as a system so new pages remain visually and technically consistent.",
  },
  {
    question: "What stack do you recommend for production?",
    answer:
      "Next.js + TypeScript + tokenized CSS with selective WebGL in hero/showcase only. Most UI effects remain pure CSS.",
  },
  {
    question: "Do you support reduced-motion accessibility?",
    answer:
      "Yes. Motion intensity automatically scales down and complex parallax or kinetic effects are disabled when users request less motion.",
  },
];

const directions: DirectionItem[] = [
  {
    id: "A",
    name: "Extreme Future Art",
    color: "Deep black + cold silver + electric cyan",
    type: "Clash Display + Inter",
    motion: "High spatial motion + hero 3D emphasis",
    brandFit: "Avant-garde tech and artistic innovation brands",
  },
  {
    id: "B",
    name: "Premium Commercial Balance",
    color: "Graphite + titanium + desaturated cyan-violet",
    type: "Sora + Manrope",
    motion: "Measured cinematic transitions with strong readability",
    brandFit: "High-budget product and enterprise innovation brands",
  },
  {
    id: "C",
    name: "Minimal Tech Luxury",
    color: "Warm charcoal + champagne silver + deep green accent",
    type: "Neue Montreal + Inter",
    motion: "Low-motion premium polish",
    brandFit: "Fintech, consulting, and luxury services",
  },
];

type VariableStyle = CSSProperties & {
  "--px"?: string;
  "--py"?: string;
  "--phase"?: string;
};

const frontendContractCompatibility = {
  use_llm: true,
  llm_segments_only: false,
  resultPath: "result.segments",
};

function App() {
  const [theme, setTheme] = useState<ThemeMode>("dark");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [contactOpen, setContactOpen] = useState(false);
  const [activeFaq, setActiveFaq] = useState<number | null>(0);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [pointer, setPointer] = useState({ x: 0, y: 0 });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const updateMotionPreference = () => {
      const prefersReduced = mediaQuery.matches;
      setReducedMotion(prefersReduced);
      if (prefersReduced) {
        setPointer({ x: 0, y: 0 });
      }
    };
    updateMotionPreference();
    mediaQuery.addEventListener("change", updateMotionPreference);
    return () => mediaQuery.removeEventListener("change", updateMotionPreference);
  }, []);

  useEffect(() => {
    const updateScrollPhase = () => {
      const maxScroll = Math.max(document.body.scrollHeight - window.innerHeight, 1);
      const phase = Math.min(window.scrollY / maxScroll, 1);
      document.documentElement.style.setProperty("--scroll-phase", phase.toFixed(3));
    };
    updateScrollPhase();
    window.addEventListener("scroll", updateScrollPhase, { passive: true });
    window.addEventListener("resize", updateScrollPhase);
    return () => {
      window.removeEventListener("scroll", updateScrollPhase);
      window.removeEventListener("resize", updateScrollPhase);
    };
  }, []);

  useEffect(() => {
    const revealTargets = document.querySelectorAll<HTMLElement>("[data-reveal]");
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
          }
        });
      },
      { threshold: 0.2, rootMargin: "0px 0px -8% 0px" },
    );
    revealTargets.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, []);

  const handleHeroMouseMove = (event: MouseEvent<HTMLElement>) => {
    if (reducedMotion) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const normalizedX = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
    const normalizedY = ((event.clientY - rect.top) / rect.height - 0.5) * 2;
    setPointer({ x: normalizedX, y: normalizedY });
  };

  const handleHeroMouseLeave = () => {
    setPointer({ x: 0, y: 0 });
  };

  const handleMagneticMove = (
    event: MouseEvent<HTMLAnchorElement | HTMLButtonElement>,
    strength: number,
  ) => {
    if (reducedMotion) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width - 0.5) * 2 * strength;
    const y = ((event.clientY - rect.top) / rect.height - 0.5) * 2 * strength;
    event.currentTarget.style.setProperty("--mx", `${x.toFixed(1)}px`);
    event.currentTarget.style.setProperty("--my", `${y.toFixed(1)}px`);
  };

  const resetMagnetic = (event: MouseEvent<HTMLAnchorElement | HTMLButtonElement>) => {
    event.currentTarget.style.setProperty("--mx", "0px");
    event.currentTarget.style.setProperty("--my", "0px");
  };

  const heroStyle = useMemo<VariableStyle>(
    () => ({
      "--px": pointer.x.toFixed(3),
      "--py": pointer.y.toFixed(3),
    }),
    [pointer],
  );

  const liquidStyle = useMemo<CSSProperties>(
    () =>
      reducedMotion
        ? {}
        : {
            transform: `translate3d(${pointer.x * 20}px, ${pointer.y * 14}px, 0) rotateX(${
              pointer.y * -7
            }deg) rotateY(${pointer.x * 10}deg)`,
          },
    [pointer, reducedMotion],
  );

  return (
    <div className="site-shell">
      <header className="top-nav glass-panel">
        <a className="brand-lockup" href="#hero">
          <span className="brand-mark" aria-hidden="true" />
          <span>
            <strong>DreamForge Studio</strong>
            <small>Liquid Experience Platform</small>
          </span>
        </a>

        <nav className="desktop-nav" aria-label="Primary">
          {navItems.map((item) => (
            <a key={item.label} href={item.href} className="nav-link">
              {item.label}
            </a>
          ))}
        </nav>

        <div className="nav-actions">
          <button
            type="button"
            className="icon-button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label="Toggle color mode"
          >
            {theme === "dark" ? "☼" : "☾"}
          </button>
          <button
            type="button"
            className="btn btn-primary magnetic"
            onClick={() => setContactOpen(true)}
            onMouseMove={(event) => handleMagneticMove(event, 5)}
            onMouseLeave={resetMagnetic}
          >
            Start Project
          </button>
          <button
            type="button"
            className="icon-button mobile-menu-button"
            aria-label="Open menu"
            onClick={() => setMobileMenuOpen(true)}
          >
            ☰
          </button>
        </div>
      </header>

      <aside className={`mobile-drawer ${mobileMenuOpen ? "open" : ""}`} aria-hidden={!mobileMenuOpen}>
        <div className="drawer-panel glass-panel">
          <div className="drawer-header">
            <strong>Navigate</strong>
            <button
              type="button"
              className="icon-button"
              onClick={() => setMobileMenuOpen(false)}
              aria-label="Close menu"
            >
              ✕
            </button>
          </div>
          <nav className="drawer-links">
            {navItems.map((item) => (
              <a key={item.label} href={item.href} onClick={() => setMobileMenuOpen(false)}>
                {item.label}
              </a>
            ))}
            <button type="button" className="btn btn-primary" onClick={() => setContactOpen(true)}>
              Start Project
            </button>
          </nav>
        </div>
        <button
          type="button"
          className="drawer-backdrop"
          aria-label="Close navigation overlay"
          onClick={() => setMobileMenuOpen(false)}
        />
      </aside>

      <main>
        <section id="hero" className="hero-section" style={heroStyle}>
          <div className="hero-copy" data-reveal>
            <p className="eyebrow">Future-Class Digital Presence</p>
            <h1>Designing Digital Gravity for ambitious brands.</h1>
            <p className="hero-subtitle">
              Liquid metal materiality, floating island composition, and cinematic interaction rhythm
              — engineered for real product velocity across desktop and mobile.
            </p>
            <div className="hero-cta">
              <button
                type="button"
                className="btn btn-primary magnetic"
                onClick={() => setContactOpen(true)}
                onMouseMove={(event) => handleMagneticMove(event, 7)}
                onMouseLeave={resetMagnetic}
              >
                Begin Premium Engagement
              </button>
              <a
                href="#cases"
                className="btn btn-secondary magnetic"
                onMouseMove={(event) => handleMagneticMove(event, 6)}
                onMouseLeave={resetMagnetic}
              >
                View Case Studies
              </a>
            </div>
            <ul className="hero-points">
              <li>High-end aesthetic with production-ready constraints</li>
              <li>Motion choreography tuned for performance and readability</li>
              <li>Distinctive brand memory without generic AI template feel</li>
            </ul>
          </div>

          <div
            className="hero-visual-stage"
            data-reveal
            onMouseMove={handleHeroMouseMove}
            onMouseLeave={handleHeroMouseLeave}
          >
            <div className="liquid-core" style={liquidStyle} aria-hidden="true">
              <div className="liquid-gloss" />
            </div>
            <article className="floating-island island-1">
              <p>Primary Promise</p>
              <strong>Future-ready with editorial clarity</strong>
            </article>
            <article className="floating-island island-2">
              <p>Motion Layer</p>
              <strong>140–320ms tactile feedback cadence</strong>
            </article>
            <article className="floating-island island-3">
              <p>Performance</p>
              <strong>Progressive enhancement by capability tier</strong>
            </article>
            <article className="floating-island island-4">
              <p>Accessibility</p>
              <strong>Reduced-motion mode fully respected</strong>
            </article>
          </div>
        </section>

        <section className="trust-strip glass-panel" data-reveal>
          <p>Trusted by teams building category-defining digital products.</p>
          <div className="trust-metrics">
            <span>+37% qualified leads</span>
            <span>2.1x demo requests</span>
            <span>98/100 UX quality score</span>
          </div>
        </section>

        <section id="about" className="content-section" data-reveal>
          <div className="section-head">
            <p className="eyebrow">About the Design Language</p>
            <h2>Balanced precision: luxury, future, technology, and art.</h2>
            <p>
              The system avoids cliché cyber aesthetics by treating material effects as supporting
              actors. Structure, contrast, and narrative hierarchy always lead.
            </p>
          </div>
          <div className="about-grid">
            <article className="glass-panel">
              <h3>Luxury</h3>
              <p>Comes from proportion, spacing, and restraint — not decorative overload.</p>
            </article>
            <article className="glass-panel">
              <h3>Future</h3>
              <p>Comes from depth cues, controlled highlights, and spatial transitions.</p>
            </article>
            <article className="glass-panel">
              <h3>Technology</h3>
              <p>Comes from crisp information architecture and interaction reliability.</p>
            </article>
            <article className="glass-panel">
              <h3>Art</h3>
              <p>Comes from composition rhythm and visual storytelling intent.</p>
            </article>
          </div>
        </section>

        <section id="services" className="content-section" data-reveal>
          <div className="section-head">
            <p className="eyebrow">Services</p>
            <h2>Complete scope beyond a landing page.</h2>
            <p>
              Designed as a full ecosystem: strategic positioning, visual system, UX rhythm, and
              implementation architecture.
            </p>
          </div>
          <div className="island-grid">
            {services.map((service) => (
              <article key={service.title} className="glass-panel service-card">
                <span>{service.tag}</span>
                <h3>{service.title}</h3>
                <p>{service.description}</p>
              </article>
            ))}
          </div>
        </section>

        <section id="features" className="content-section" data-reveal>
          <div className="section-head">
            <p className="eyebrow">Feature Deep Dive</p>
            <h2>Every effect has a job, every section has value.</h2>
          </div>
          <div className="feature-list">
            {features.map((feature) => (
              <article key={feature.title} className="glass-panel feature-card">
                <div>
                  <h3>{feature.title}</h3>
                  <p>{feature.body}</p>
                </div>
                <strong>{feature.metric}</strong>
              </article>
            ))}
          </div>
        </section>

        <section id="showcase" className="content-section" data-reveal>
          <div className="section-head">
            <p className="eyebrow">Visual Showcase</p>
            <h2>Material, typography, interaction, and depth in one cohesive system.</h2>
          </div>
          <div className="showcase-grid">
            {showcaseItems.map((item) => (
              <article key={item.title} className="showcase-card glass-panel">
                <span>{item.label}</span>
                <h3>{item.title}</h3>
              </article>
            ))}
          </div>
        </section>

        <section id="cases" className="content-section" data-reveal>
          <div className="section-head">
            <p className="eyebrow">Case Studies</p>
            <h2>Design outcomes tied to business signals.</h2>
          </div>
          <div className="case-grid">
            {caseStudies.map((item) => (
              <article key={item.client} className="glass-panel case-card">
                <h3>{item.client}</h3>
                <p>
                  <strong>Challenge:</strong> {item.challenge}
                </p>
                <p>
                  <strong>Outcome:</strong> {item.outcome}
                </p>
              </article>
            ))}
          </div>
        </section>

        <section className="content-section" data-reveal>
          <div className="section-head">
            <p className="eyebrow">Process</p>
            <h2>Structured, premium delivery from concept to launch.</h2>
          </div>
          <ol className="process-list">
            {processSteps.map((step) => (
              <li key={step.title} className="glass-panel">
                <h3>{step.title}</h3>
                <p>{step.body}</p>
              </li>
            ))}
          </ol>
        </section>

        <section className="content-section" data-reveal>
          <div className="section-head">
            <p className="eyebrow">Design Directions</p>
            <h2>Three production-ready variants with distinct brand fit.</h2>
            <p className="recommend-note">
              Recommended: <strong>Direction B — Premium Commercial Balance</strong>.
            </p>
          </div>
          <div className="direction-grid">
            {directions.map((direction) => (
              <article key={direction.id} className="glass-panel direction-card">
                <header>
                  <span>Direction {direction.id}</span>
                  {direction.id === "B" && <em>Recommended</em>}
                </header>
                <h3>{direction.name}</h3>
                <p>
                  <strong>Color:</strong> {direction.color}
                </p>
                <p>
                  <strong>Type:</strong> {direction.type}
                </p>
                <p>
                  <strong>Motion:</strong> {direction.motion}
                </p>
                <p>
                  <strong>Best fit:</strong> {direction.brandFit}
                </p>
              </article>
            ))}
          </div>
        </section>

        <section id="faq" className="content-section" data-reveal>
          <div className="section-head">
            <p className="eyebrow">FAQ</p>
            <h2>Answers that reduce decision friction.</h2>
          </div>
          <div className="faq-list">
            {faqs.map((item, index) => {
              const expanded = activeFaq === index;
              return (
                <article key={item.question} className={`glass-panel faq-item ${expanded ? "open" : ""}`}>
                  <button
                    type="button"
                    onClick={() => setActiveFaq(expanded ? null : index)}
                    aria-expanded={expanded}
                  >
                    <span>{item.question}</span>
                    <span>{expanded ? "−" : "+"}</span>
                  </button>
                  <p>{item.answer}</p>
                </article>
              );
            })}
          </div>
        </section>

        <section className="cta-section glass-panel" data-reveal>
          <p className="eyebrow">Final CTA</p>
          <h2>Build a website that feels inevitable, not merely impressive.</h2>
          <p>
            A premium digital presence should look distinctive, read clearly, and convert with
            confidence on every device.
          </p>
          <button
            type="button"
            className="btn btn-primary magnetic"
            onClick={() => setContactOpen(true)}
            onMouseMove={(event) => handleMagneticMove(event, 8)}
            onMouseLeave={resetMagnetic}
          >
            Book Discovery Session
          </button>
        </section>
      </main>

      <footer
        className="site-footer"
        data-contract-use-llm={String(frontendContractCompatibility.use_llm)}
        data-contract-llm-segments-only={String(frontendContractCompatibility.llm_segments_only)}
        data-contract-result-path={frontendContractCompatibility.resultPath}
      >
        <div>
          <strong>DreamForge Studio</strong>
          <p>High-end digital product experience, from strategy to build.</p>
        </div>
        <div className="footer-links">
          <a href="#hero">Top</a>
          <a href="#services">Services</a>
          <a href="#cases">Cases</a>
          <button type="button" onClick={() => setContactOpen(true)}>
            Contact
          </button>
        </div>
      </footer>

      <section className={`modal-layer ${contactOpen ? "open" : ""}`} aria-hidden={!contactOpen}>
        <button
          type="button"
          className="modal-backdrop"
          onClick={() => setContactOpen(false)}
          aria-label="Close contact modal"
        />
        <div className="modal-panel glass-panel" role="dialog" aria-modal="true" aria-label="Start project">
          <header>
            <h3>Start a High-end Project</h3>
            <button type="button" className="icon-button" onClick={() => setContactOpen(false)}>
              ✕
            </button>
          </header>
          <form>
            <label>
              Name
              <input type="text" placeholder="Your name" />
            </label>
            <label>
              Work Email
              <input type="email" placeholder="name@company.com" />
            </label>
            <label>
              Project Focus
              <textarea
                rows={4}
                placeholder="Tell us your goals, scope, and timeline."
              />
            </label>
            <div className="modal-actions">
              <a className="btn btn-secondary" href="mailto:hello@dreamforge.ai">
                Send via Email
              </a>
              <button type="button" className="btn btn-primary" onClick={() => setContactOpen(false)}>
                Save Inquiry Draft
              </button>
            </div>
          </form>
        </div>
      </section>
    </div>
  );
}

export default App;
