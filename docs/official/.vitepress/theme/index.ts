import { h, nextTick, watch } from 'vue'
import type { Theme } from 'vitepress'
import { useData } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import { createMermaidRenderer } from 'vitepress-mermaid-renderer'
import './style.css'

const rawContentBase = 'https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/'
const agentDocsUrl = `${rawContentBase}agents.md`

export default {
  extends: DefaultTheme,
  Layout: () => {
    const { isDark, page } = useData()

    const RawMarkdownLink = () => {
      if (page.value.isNotFound || !page.value.relativePath) {
        return null
      }

      const rawUrl = `${rawContentBase}${page.value.relativePath}`

      return h('div', { class: 'raw-markdown-link' }, [
        h(
          'a',
          {
            href: rawUrl,
            target: '_blank',
            rel: 'noopener',
            'data-raw-markdown-link': page.value.relativePath
          },
          'View raw Markdown'
        )
      ])
    }

    const AgentDocsBanner = () => {
      if (page.value.isNotFound || page.value.relativePath !== 'index.md') {
        return null
      }

      return h('div', { class: 'agent-docs-banner' }, [
        h('span', { class: 'agent-docs-banner__label' }, 'Agent docs'),
        h('span', { class: 'agent-docs-banner__text' }, 'If you are an agent, follow this link for raw documentation.'),
        h(
          'a',
          {
            href: agentDocsUrl,
            target: '_blank',
            rel: 'noopener'
          },
          'Open raw docs'
        )
      ])
    }

    const renderMermaid = () => {
      createMermaidRenderer({
        theme: isDark.value ? 'dark' : 'default',
        startOnLoad: false,
        flowchart: {
          useMaxWidth: true,
          htmlLabels: true
        }
      })
    }

    nextTick(renderMermaid)
    watch(() => isDark.value, renderMermaid)

    return h(DefaultTheme.Layout, null, {
      'doc-before': RawMarkdownLink,
      'home-hero-before': AgentDocsBanner,
      'home-features-after': RawMarkdownLink
    })
  }
} satisfies Theme
