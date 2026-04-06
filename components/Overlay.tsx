"use client";

import { useEffect, useRef, useState } from "react";
import {
  motion,
  useScroll,
  useTransform,
  MotionValue,
} from "framer-motion";

// ─── Types ──────────────────────────────────────────────────────
interface OverlaySection {
  tag: string;
  title: string;
  subtitle: string;
  /** scroll progress range [enter, peak-start, peak-end, exit] */
  range: [number, number, number, number];
  align: "left" | "center" | "right";
}

// ─── Section Config ─────────────────────────────────────────────
const SECTIONS: OverlaySection[] = [
  {
    tag: "// init sequence",
    title: "ExoVision AI",
    subtitle: "Discover New Worlds",
    range: [0.0, 0.03, 0.14, 0.22],
    align: "center",
  
    
  },
  {
    tag: "// module::ingest",
    title: "Upload Stellar Data",
    subtitle: "Analyze Light Curves",
    range: [0.25, 0.30, 0.40, 0.48],
    align: "left",
  },
  {
    tag: "// module::detect",
    title: "Detect Exoplanets",
    subtitle: "Reveal Hidden Signals",
    range: [0.52, 0.58, 0.68, 0.76],
    align: "right",
  },
];

// ─── Animated Section ───────────────────────────────────────────
function AnimatedSection({
  section,
  scrollYProgress,
}: {
  section: OverlaySection;
  scrollYProgress: MotionValue<number>;
}) {
  const { range, align, tag, title, subtitle } = section;
  const [enter, peakStart, peakEnd, exit] = range;

  // Opacity: fade in → hold → fade out
  const opacity = useTransform(
    scrollYProgress,
    [enter, peakStart, peakEnd, exit],
    [0, 1, 1, 0]
  );

  // Vertical parallax
  const y = useTransform(
    scrollYProgress,
    [enter, peakStart, peakEnd, exit],
    [80, 0, 0, -60]
  );

  // Horizontal slide based on alignment
  const xDirection = align === "right" ? 1 : align === "left" ? -1 : 0;
  const x = useTransform(
    scrollYProgress,
    [enter, peakStart, peakEnd, exit],
    [60 * xDirection, 0, 0, -40 * xDirection]
  );

  // Blur → sharp → blur
  const blurValue = useTransform(
    scrollYProgress,
    [enter, peakStart, peakEnd, exit],
    [12, 0, 0, 8]
  );
  const filter = useTransform(blurValue, (v) => `blur(${v}px)`);

  // Scale
  const scale = useTransform(
    scrollYProgress,
    [enter, peakStart, peakEnd, exit],
    [0.92, 1, 1, 0.96]
  );

  // Tag line opacity (slightly delayed)
  const tagOpacity = useTransform(
    scrollYProgress,
    [enter, peakStart + 0.01, peakEnd - 0.01, exit],
    [0, 1, 1, 0]
  );

  // Subtitle Y offset (slightly delayed entrance)
  const subtitleY = useTransform(
    scrollYProgress,
    [enter + 0.01, peakStart + 0.02, peakEnd, exit],
    [30, 0, 0, -20]
  );

  const subtitleOpacity = useTransform(
    scrollYProgress,
    [enter + 0.01, peakStart + 0.02, peakEnd, exit],
    [0, 1, 1, 0]
  );

  // Alignment classes
  const alignClass =
    align === "left"
      ? "items-start text-left pl-8 sm:pl-16 md:pl-24 lg:pl-32"
      : align === "right"
      ? "items-end text-right pr-8 sm:pr-16 md:pr-24 lg:pr-32"
      : "items-center text-center";

  return (
    <motion.div
      className={`fixed inset-0 z-10 flex flex-col justify-center ${alignClass} pointer-events-none`}
      style={{ opacity }}
    >
      <motion.div
        style={{ y, x, filter, scale }}
        className="flex flex-col gap-4 sm:gap-5 md:gap-6 max-w-2xl"
      >
        {/* Tag Line */}
        <motion.span className="overlay-tag w-fit" style={{ opacity: tagOpacity }}>
          {tag}
        </motion.span>

        {/* Title */}
        <h2 className="overlay-title text-4xl sm:text-5xl md:text-6xl lg:text-7xl xl:text-8xl">
          {title}
        </h2>

        {/* Subtitle */}
        <motion.p
          className="overlay-subtitle text-sm sm:text-base md:text-lg lg:text-xl"
          style={{ y: subtitleY, opacity: subtitleOpacity }}
        >
          {subtitle}
        </motion.p>

        {/* Decorative line */}
        <motion.div
          className="h-px w-16 sm:w-20 md:w-24"
          style={{
            opacity: subtitleOpacity,
            background:
              "linear-gradient(90deg, var(--accent-primary), var(--accent-secondary), transparent)",
          }}
        />
      </motion.div>
    </motion.div>
  );
}

// ─── Scroll Progress Indicator ──────────────────────────────────
function ScrollIndicator({
  scrollYProgress,
}: {
  scrollYProgress: MotionValue<number>;
}) {
  const indicatorOpacity = useTransform(
    scrollYProgress,
    [0, 0.02, 0.85, 0.95],
    [0.6, 0.6, 0.6, 0]
  );

  const scaleX = useTransform(scrollYProgress, [0, 1], [0, 1]);

  return (
    <motion.div
      className="fixed bottom-0 left-0 right-0 z-20 pointer-events-none"
      style={{ opacity: indicatorOpacity }}
    >
      <div className="h-[2px] w-full bg-white/[0.04]">
        <motion.div
          className="h-full origin-left"
          style={{
            scaleX,
            background:
              "linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))",
            boxShadow: "0 0 20px var(--glow-primary)",
          }}
        />
      </div>
    </motion.div>
  );
}

// ─── Scroll Down Prompt ─────────────────────────────────────────
function ScrollPrompt({
  scrollYProgress,
}: {
  scrollYProgress: MotionValue<number>;
}) {
  const promptOpacity = useTransform(
    scrollYProgress,
    [0, 0.01, 0.06, 0.10],
    [0, 1, 1, 0]
  );

  const promptY = useTransform(
    scrollYProgress,
    [0, 0.01, 0.06, 0.10],
    [20, 0, 0, -20]
  );

  return (
    <motion.div
      className="fixed bottom-12 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center gap-3 pointer-events-none"
      style={{ opacity: promptOpacity, y: promptY }}
    >
      <span
        className="text-xs tracking-[0.2em] uppercase"
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          color: "var(--text-muted)",
        }}
      >
        Scroll to Explore
      </span>
      <motion.div
        className="w-5 h-8 rounded-full border border-white/20 flex items-start justify-center pt-1.5"
        animate={{ opacity: [0.3, 0.8, 0.3] }}
        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
      >
        <motion.div
          className="w-1 h-1.5 rounded-full bg-white/60"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        />
      </motion.div>
    </motion.div>
  );
}

// ─── Main Overlay Component ─────────────────────────────────────
export default function Overlay() {
  const targetRef = useRef<HTMLDivElement | null>(null);
  const [mounted, setMounted] = useState(false);

  // Find the scroll container after mount
  useEffect(() => {
    const el = document.getElementById("scrolly-container");
    if (el) {
      targetRef.current = el as HTMLDivElement;
      setMounted(true);
    }
  }, []);

  const { scrollYProgress } = useScroll({
    target: mounted ? targetRef : undefined,
    offset: ["start start", "end end"],
  });

  if (!mounted) return null;

  return (
    <>
      {SECTIONS.map((section, i) => (
        <AnimatedSection
          key={i}
          section={section}
          scrollYProgress={scrollYProgress}
        />
      ))}
      <ScrollIndicator scrollYProgress={scrollYProgress} />
      <ScrollPrompt scrollYProgress={scrollYProgress} />
    </>
  );
}
