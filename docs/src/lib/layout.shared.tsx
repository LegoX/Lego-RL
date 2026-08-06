import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <div className="flex items-center gap-2 mr-4">
          <img
            src="/legorl_emblem.png"
            alt=""
            width={24}
            height={24}
            style={{ width: 24, height: 24 }}
          />
          <p className="docs-brand-title">
            Lego-<span className="docs-rl-word">RL</span>
          </p>
        </div>
      ),
    },
    links: [
      {
        url: "/docs",
        text: "docs",
        active: "nested-url",
      },
      {
        url: "/docs/dashboard",
        text: "dashboard",
        active: "nested-url",
      },
    ],
    themeSwitch: {
      enabled: true,
      mode: "light-dark-system",
    },
  };
}
