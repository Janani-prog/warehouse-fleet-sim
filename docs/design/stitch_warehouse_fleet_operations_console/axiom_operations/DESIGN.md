---
name: Axiom Operations
colors:
  surface: '#f9f9f9'
  surface-dim: '#dadada'
  surface-bright: '#f9f9f9'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f3f3f3'
  surface-container: '#eeeeee'
  surface-container-high: '#e8e8e8'
  surface-container-highest: '#e2e2e2'
  on-surface: '#1a1c1c'
  on-surface-variant: '#424656'
  inverse-surface: '#2f3131'
  inverse-on-surface: '#f0f1f1'
  outline: '#727687'
  outline-variant: '#c2c6d8'
  surface-tint: '#0054d6'
  primary: '#0050cb'
  on-primary: '#ffffff'
  primary-container: '#0066ff'
  on-primary-container: '#f8f7ff'
  inverse-primary: '#b3c5ff'
  secondary: '#5f5e5e'
  on-secondary: '#ffffff'
  secondary-container: '#e5e2e1'
  on-secondary-container: '#656464'
  tertiary: '#a33200'
  on-tertiary: '#ffffff'
  tertiary-container: '#cc4204'
  on-tertiary-container: '#fff6f4'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dae1ff'
  primary-fixed-dim: '#b3c5ff'
  on-primary-fixed: '#001849'
  on-primary-fixed-variant: '#003fa4'
  secondary-fixed: '#e5e2e1'
  secondary-fixed-dim: '#c8c6c5'
  on-secondary-fixed: '#1c1b1b'
  on-secondary-fixed-variant: '#474646'
  tertiary-fixed: '#ffdbd0'
  tertiary-fixed-dim: '#ffb59d'
  on-tertiary-fixed: '#390c00'
  on-tertiary-fixed-variant: '#832600'
  background: '#f9f9f9'
  on-background: '#1a1c1c'
  surface-variant: '#e2e2e2'
typography:
  headline-lg:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.01em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  label-mono:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  button:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 16px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  container-padding: 24px
  gutter: 16px
  sidebar-width: 240px
  row-height-sm: 32px
  row-height-md: 44px
---

## Brand & Style

The design system is rooted in functional minimalism and engineering precision. It is designed for high-density internal operations where clarity, speed of scanning, and reduced cognitive load are paramount. 

The aesthetic is "quiet" and utilitarian, drawing inspiration from developer-centric tools. It avoids all decorative flourishes, relying instead on rigorous alignment, generous whitespace, and a strict typographic scale to establish hierarchy. The emotional response is one of calm efficiency and technical reliability.

## Colors

The palette is strictly functional. 
- **Backgrounds:** Use `#FAFAFA` for the primary canvas. Use white (`#FFFFFF`) for elevated surfaces like cards or active sidebar items to create subtle contrast.
- **Typography:** Primary text is `#111111`. Secondary or metadata text uses `#666666`.
- **Borders:** A universal `#E5E5E5` is used for all structural divisions, table rows, and component strokes.
- **Accent:** `#0066FF` is reserved exclusively for interactive "Primary" actions, active navigation states, and critical data thresholds. It should never be used decoratively.
- **Status:** Use a semantic green for success and red for errors, but keep them desaturated to maintain the neutral aesthetic.

## Typography

This design system utilizes **Inter** for all UI elements to ensure maximum legibility and a neutral tone. **JetBrains Mono** is introduced specifically for timestamps, IDs, and tabular data to ensure character alignment and a technical feel.

- **Constraint:** Limit font weights to 400 (Regular), 500 (Medium), and 600 (Semi-Bold). 
- **Scale:** Keep sizes small. The maximum headline size should rarely exceed 20px to maintain data density.
- **Tracking:** Apply slight negative letter-spacing to headlines for a tighter, more professional appearance.

## Layout & Spacing

The layout follows a strict **4px baseline grid**. 

- **Structure:** A fixed left sidebar (`240px`) anchors the navigation. The main content area uses a fluid width with a maximum max-width of `1440px` for readability, centered on the screen.
- **Density:** Use tight vertical spacing in data tables (`32px` or `44px` row heights) to maximize information density.
- **Margins:** Maintain a consistent `24px` padding around major page sections.
- **Alignment:** All elements must align to the left. Avoid center-alignment except for icons or specific small-scale badges.

## Elevation & Depth

This design system is **flat**. There are no shadows, blurs, or gradients.

- **Hierarchy through Borders:** Use 1px solid `#E5E5E5` lines to separate the sidebar, top bar, and content sections.
- **Hierarchy through Fills:** Use subtle shifts in background color (e.g., `#FFFFFF` vs `#FAFAFA`) to indicate focus or nesting.
- **Interactive States:** Use a 1px border or a subtle neutral fill change (e.g., `#F0F0F0`) for hover states. Shadows are strictly prohibited.

## Shapes

Shapes are disciplined and "Soft" (4px radius). This provides a hint of approachability without feeling overly consumer-oriented.

- **Buttons & Inputs:** 4px border-radius.
- **Badges/Chips:** 2px or 4px radius; never pill-shaped.
- **Selection Indicators:** Sidebar active states should use a 2px vertical "pill" or line aligned to the far left or right of the menu item.

## Components

- **Sidebar:** Navigation items use 16px line icons with 13px Inter text. Active state uses a subtle `#F4F4F4` background and primary accent text color.
- **KPI Strips:** Minimalist cards with a 1px border. The value is 20px Semi-Bold, and the label is 12px Mono Muted. No background colors.
- **Data Tables:** Headers are 12px Semi-Bold, all-caps with 0.05em tracking. Rows are separated by 1px horizontal lines only. Use JetBrains Mono for all numeric values and timestamps.
- **Input Fields:** 1px `#E5E5E5` border, 4px radius. On focus, the border changes to the Primary Accent color. No inner shadows.
- **Buttons:** 
  - *Primary:* Accent background, white text. 
  - *Secondary:* White background, 1px border, primary text.
  - *Ghost:* No border, primary text, subtle gray fill on hover.
- **Charts:** Line charts use a 1.5px stroke width in the Primary Accent color. Tooltips are flat, white boxes with 1px borders and no shadows.