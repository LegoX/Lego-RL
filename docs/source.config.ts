import {
  defineConfig,
  defineDocs,
  frontmatterSchema,
  metaSchema,
} from 'fumadocs-mdx/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    schema: frontmatterSchema,
  },
  meta: {
    schema: metaSchema,
  },
});

export default defineConfig({
  mdxOptions: {
    // KaTeX: $…$ inline, $$…$$ display. remark-math must run before the
    // fumadocs defaults, rehype-katex after them.
    remarkPlugins: (v) => [remarkMath, ...v],
    rehypePlugins: (v) => [rehypeKatex, ...v],
  },
});
