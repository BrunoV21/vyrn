import { defineConfig } from 'vitepress'

const siteBase = '/vyrn/'

export default defineConfig({
  title: 'vyrn',
  description: 'A token-efficient Rust CLI agent for local and small LLMs.',
  base: siteBase,
  appearance: 'force-dark',

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${siteBase}favicon.svg` }],
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', { href: 'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap', rel: 'stylesheet' }]
  ],

  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Get Started', link: '/getting-started/' },
      { text: 'Commands', link: '/commands/' },
      { text: 'Concepts', link: '/concepts/architecture' },
      { text: 'Releases', link: '/releases/' },
      { text: 'Community', link: '/community' },
      { text: 'Roadmap', link: '/roadmap' }
    ],

    sidebar: [
      {
        text: '// GETTING STARTED',
        items: [
          { text: 'Overview', link: '/getting-started/' },
          { text: 'Installation', link: '/getting-started/installation' },
          { text: 'First Run', link: '/getting-started/first-run' }
        ]
      },
      {
        text: '// COMMANDS',
        items: [
          { text: 'Overview', link: '/commands/' },
          { text: 'Session Options', link: '/commands/session' },
          { text: 'Slash Commands', link: '/commands/slash-commands' }
        ]
      },
      {
        text: '// CONCEPTS',
        items: [
          { text: 'Architecture', link: '/concepts/architecture' },
          { text: 'Context Management', link: '/concepts/context-management' },
          { text: 'Core Tools', link: '/concepts/core-tools' },
          { text: 'Skills and MCP', link: '/concepts/skills-and-mcp' },
          { text: 'Token Savings', link: '/concepts/token-savings' }
        ]
      },
      {
        text: '// MORE',
        items: [
          { text: 'Releases', link: '/releases/' },
          { text: 'Community', link: '/community' },
          { text: 'Roadmap', link: '/roadmap' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/BrunoV21/vyrn' }
    ],

    footer: {
      message: 'vyrn - token-efficient agent infrastructure for local-first Rust users'
    },

    search: {
      provider: 'local'
    }
  }
})
