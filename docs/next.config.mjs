import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  // Static export so the site can be served from Cloudflare Pages as plain files.
  // Note: `output: 'export'` disables next.config redirects() and image
  // optimization. The root redirect (/ -> /docs) is expressed in public/_redirects.
  output: 'export',
  images: { unoptimized: true },
};

export default withMDX(config);
