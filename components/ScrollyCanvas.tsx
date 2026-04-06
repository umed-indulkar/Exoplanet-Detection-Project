"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useScroll, useTransform, useMotionValueEvent } from "framer-motion";

// ─── Configuration ──────────────────────────────────────────────
const TOTAL_FRAMES = 192;
const SCROLL_HEIGHT_VH = 500; // how tall the scroll container is

function getFramePath(index: number): string {
  const padded = String(index).padStart(3, "0");
  return `/sequence/frame_${padded}_delay-0.041s.webp`;
}

// ─── Cover-fit drawing (emulates object-fit: cover) ─────────────
function drawCover(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  canvasW: number,
  canvasH: number
) {
  const imgRatio = img.naturalWidth / img.naturalHeight;
  const canvasRatio = canvasW / canvasH;

  let drawW: number, drawH: number, offsetX: number, offsetY: number;

  if (canvasRatio > imgRatio) {
    // canvas is wider → fit width, crop height
    drawW = canvasW;
    drawH = canvasW / imgRatio;
    offsetX = 0;
    offsetY = (canvasH - drawH) / 2;
  } else {
    // canvas is taller → fit height, crop width
    drawH = canvasH;
    drawW = canvasH * imgRatio;
    offsetX = (canvasW - drawW) / 2;
    offsetY = 0;
  }

  ctx.drawImage(img, offsetX, offsetY, drawW, drawH);
}

// ─── Component ──────────────────────────────────────────────────
export default function ScrollyCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imagesRef = useRef<HTMLImageElement[]>([]);
  const currentFrameRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const [loadProgress, setLoadProgress] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);

  // ─── Scroll tracking ───────────────────────────────────────────
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"],
  });

  const frameIndex = useTransform(
    scrollYProgress,
    [0, 1],
    [0, TOTAL_FRAMES - 1]
  );

  // ─── Render a frame to canvas ─────────────────────────────────
  const renderFrame = useCallback((index: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;

    const img = imagesRef.current[index];
    if (!img || !img.complete || img.naturalWidth === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const displayW = canvas.clientWidth;
    const displayH = canvas.clientHeight;

    // Only resize the backing store if needed
    if (canvas.width !== displayW * dpr || canvas.height !== displayH * dpr) {
      canvas.width = displayW * dpr;
      canvas.height = displayH * dpr;
      ctx.scale(dpr, dpr);
    }

    ctx.clearRect(0, 0, displayW, displayH);
    drawCover(ctx, img, displayW, displayH);
  }, []);

  // ─── Listen to scroll and render ──────────────────────────────
  useMotionValueEvent(frameIndex, "change", (latest) => {
    const idx = Math.round(latest);
    if (idx === currentFrameRef.current) return;
    currentFrameRef.current = idx;

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      renderFrame(idx);
    });
  });

  // ─── Preload all images ───────────────────────────────────────
  useEffect(() => {
    let loaded = 0;
    const images: HTMLImageElement[] = new Array(TOTAL_FRAMES);

    const onLoad = () => {
      loaded++;
      setLoadProgress(loaded / TOTAL_FRAMES);
      if (loaded === TOTAL_FRAMES) {
        imagesRef.current = images;
        setIsLoaded(true);
        // Draw the first frame
        renderFrame(0);
      }
    };

    for (let i = 0; i < TOTAL_FRAMES; i++) {
      const img = new Image();
      img.src = getFramePath(i);
      img.onload = onLoad;
      img.onerror = onLoad; // count errors too so we don't hang
      images[i] = img;
    }

    return () => {
      // cleanup
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [renderFrame]);

  // ─── Handle resize ────────────────────────────────────────────
  useEffect(() => {
    const handleResize = () => {
      renderFrame(currentFrameRef.current);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [renderFrame]);

  return (
    <>
      {/* Loading Screen */}
      <div className={`loading-screen ${isLoaded ? "loaded" : ""}`}>
        <div className="flex flex-col items-center gap-4">
          <h2
            className="text-lg tracking-[0.2em] uppercase"
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              color: "var(--text-secondary)",
            }}
          >
            ExoVision AI
          </h2>
          <div className="loading-bar-track">
            <div
              className="loading-bar-fill"
              style={{ width: `${loadProgress * 100}%` }}
            />
          </div>
          <span
            className="text-xs"
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              color: "var(--text-muted)",
            }}
          >
            {Math.round(loadProgress * 100)}% — Loading Stellar Data
          </span>
        </div>
      </div>

      {/* Scroll Container */}
      <div
        ref={containerRef}
        id="scrolly-container"
        style={{ height: `${SCROLL_HEIGHT_VH}vh` }}
        className="relative"
      >
        {/* Sticky Canvas */}
        <div className="sticky top-0 h-screen w-full overflow-hidden">
          <canvas
            ref={canvasRef}
            className="block h-full w-full"
            style={{ imageRendering: "auto" }}
          />
          {/* Vignette Overlay */}
          <div
            className="pointer-events-none absolute inset-0"
            style={{
              background: `
                radial-gradient(ellipse at center, transparent 40%, rgba(3,0,20,0.5) 100%),
                linear-gradient(to bottom, rgba(3,0,20,0.3) 0%, transparent 15%, transparent 85%, rgba(3,0,20,0.6) 100%)
              `,
            }}
          />
        </div>
      </div>
    </>
  );
}
