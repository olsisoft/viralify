'use client';

import { useEffect, useRef } from 'react';

export function PerformanceChart() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * 2;
    canvas.height = rect.height * 2;
    ctx.scale(2, 2);

    const width = rect.width;
    const height = rect.height;

    // Demo data - views over last 7 days
    const data = [45000, 52000, 48000, 71000, 68000, 89000, 125000];
    const labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

    const maxValue = Math.max(...data) * 1.2;
    const minValue = 0;

    // Calculate points
    const padding = { top: 20, right: 20, bottom: 30, left: 50 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    const points = data.map((value, index) => ({
      x: padding.left + (index * chartWidth) / (data.length - 1),
      y: padding.top + chartHeight - ((value - minValue) / (maxValue - minValue)) * chartHeight,
    }));

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (chartHeight / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    // Draw gradient fill
    const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    gradient.addColorStop(0, 'rgba(254, 44, 85, 0.3)');
    gradient.addColorStop(1, 'rgba(254, 44, 85, 0)');

    ctx.beginPath();
    ctx.moveTo(points[0].x, height - padding.bottom);
    points.forEach((point) => ctx.lineTo(point.x, point.y));
    ctx.lineTo(points[points.length - 1].x, height - padding.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw line
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.forEach((point, index) => {
      if (index > 0) {
        ctx.lineTo(point.x, point.y);
      }
    });
    ctx.strokeStyle = '#fe2c55';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw points
    points.forEach((point) => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#fe2c55';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // Draw labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    labels.forEach((label, index) => {
      const x = padding.left + (index * chartWidth) / (data.length - 1);
      ctx.fillText(label, x, height - 8);
    });

    // Draw Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const value = minValue + ((maxValue - minValue) / 4) * (4 - i);
      const y = padding.top + (chartHeight / 4) * i;
      const label = value >= 1000 ? `${(value / 1000).toFixed(0)}K` : value.toString();
      ctx.fillText(label, padding.left - 8, y + 4);
    }
  }, []);

  return (
    <div className="w-full h-64">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
