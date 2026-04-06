"use client";

import { motion } from "framer-motion";

const FEATURES = [
  {
    icon: "◈",
    title: "Spectral Analysis",
    description: "AI-driven light curve decomposition with 99.7% accuracy across stellar classifications.",
  },
  {
    icon: "◇",
    title: "Transit Detection",
    description: "Real-time photometric transit identification using deep neural architectures.",
  },
  {
    icon: "△",
    title: "Habitability Index",
    description: "Multi-parameter habitability scoring based on atmospheric and orbital dynamics.",
  },
];

export default function NextSection() {
  return (
    <section className="relative min-h-screen next-section-gradient overflow-hidden">
      {/* Background grid */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `
            linear-gradient(rgba(124,58,237,0.3) 1px, transparent 1px),
            linear-gradient(90deg, rgba(124,58,237,0.3) 1px, transparent 1px)
          `,
          backgroundSize: "60px 60px",
        }}
      />

      <div className="relative z-10 w-full px-6 sm:px-10 md:px-16 lg:px-20 xl:px-28 py-24 sm:py-32 md:py-40">
        {/* Section Header */}
        <motion.div
          className="flex flex-col items-center text-center mb-16 sm:mb-20 md:mb-24"
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        >
          <span className="overlay-tag mb-6">// next::frontier</span>
          <h2
            className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight"
            style={{
              fontFamily: "'Space Grotesk', system-ui, sans-serif",
              background: "linear-gradient(135deg, #ffffff 0%, rgba(255,255,255,0.5) 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            Beyond the Observable
          </h2>
          <p
            className="mt-4 sm:mt-6 max-w-lg text-sm sm:text-base"
            style={{ color: "var(--text-secondary)" }}
          >
            Pushing the boundaries of exoplanetary science with next-generation AI inference engines.
          </p>
        </motion.div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
          {FEATURES.map((feature, i) => (
            <motion.div
              key={i}
              className="glass-panel p-6 sm:p-8 flex flex-col gap-4 group cursor-default"
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{
                duration: 0.7,
                delay: i * 0.15,
                ease: [0.16, 1, 0.3, 1],
              }}
              whileHover={{ y: -4, transition: { duration: 0.3 } }}
            >
              {/* Icon */}
              <div
                className="w-12 h-12 flex items-center justify-center rounded-xl text-xl"
                style={{
                  background: "rgba(124, 58, 237, 0.08)",
                  border: "1px solid rgba(124, 58, 237, 0.15)",
                  color: "var(--accent-primary)",
                }}
              >
                {feature.icon}
              </div>

              {/* Title */}
              <h3
                className="text-lg sm:text-xl font-semibold"
                style={{
                  fontFamily: "'Space Grotesk', system-ui, sans-serif",
                  color: "var(--text-primary)",
                }}
              >
                {feature.title}
              </h3>

              {/* Description */}
              <p
                className="text-sm leading-relaxed"
                style={{ color: "var(--text-secondary)" }}
              >
                {feature.description}
              </p>

              {/* Bottom accent */}
              <div
                className="mt-auto h-px w-full transition-all duration-500 group-hover:w-full"
                style={{
                  background:
                    "linear-gradient(90deg, var(--accent-primary), var(--accent-secondary), transparent)",
                  opacity: 0.3,
                }}
              />
            </motion.div>
          ))}
        </div>

        {/* Launch AI Lab CTA */}
        <motion.div
          className="mt-16 sm:mt-20 md:mt-24 flex flex-col items-center gap-6"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
        >
          <a
            href="http://localhost:8501"
            target="_blank"
            rel="noopener noreferrer"
            className="group inline-flex items-center gap-3 px-8 py-4 rounded-2xl text-sm font-semibold tracking-wide transition-all duration-300"
            style={{
              fontFamily: "'Space Grotesk', system-ui, sans-serif",
              background: "linear-gradient(135deg, rgba(0,229,255,0.12), rgba(124,58,237,0.12))",
              border: "1px solid rgba(0,229,255,0.25)",
              color: "#00e5ff",
              boxShadow: "0 0 30px rgba(0,229,255,0.08)",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = "rgba(0,229,255,0.5)";
              e.currentTarget.style.boxShadow = "0 0 50px rgba(0,229,255,0.15), 0 0 100px rgba(124,58,237,0.08)";
              e.currentTarget.style.transform = "translateY(-2px)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = "rgba(0,229,255,0.25)";
              e.currentTarget.style.boxShadow = "0 0 30px rgba(0,229,255,0.08)";
              e.currentTarget.style.transform = "translateY(0)";
            }}
          >
            <span style={{ fontSize: "1.2rem" }}>🧪</span>
            Launch AI Lab
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"
              style={{ opacity: 0.6 }}>
              <path d="M3 13L13 3M13 3H5M13 3V11" stroke="currentColor" strokeWidth="1.5"
                strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </a>
          <div
            className="flex items-center gap-3 text-xs tracking-[0.15em] uppercase"
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              color: "var(--text-muted)",
            }}
          >
            <span className="inline-block w-8 h-px bg-white/10" />
            Streamlit Dashboard
            <span className="inline-block w-8 h-px bg-white/10" />
          </div>
        </motion.div>
      </div>
    </section>
  );
}
