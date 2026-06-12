<script setup>
const startMarker = '<!-- release-notes:start -->'
const endMarker = '<!-- release-notes:end -->'

const releaseFiles = import.meta.glob('../../releases/v*.md', {
  query: '?raw',
  import: 'default',
  eager: true
})

function extractTag(file) {
  return file.split('/').pop().replace(/\.md$/, '')
}

function extractTitle(source, tag) {
  const titleMatch = source.match(/^title:\s*(.+)$/m)
  return titleMatch ? titleMatch[1].trim().replace(/^["']|["']$/g, '') : `Vyrn ${tag}`
}

function extractReleaseBody(source, file) {
  const start = source.indexOf(startMarker)
  const end = source.indexOf(endMarker)

  if (start === -1 || end === -1 || end <= start) {
    throw new Error(`${file} must contain release-notes start and end markers`)
  }

  const body = source.slice(start + startMarker.length, end).trim()
  if (!body) {
    throw new Error(`${file} must contain release notes between markers`)
  }

  return body
}

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function renderInline(value) {
  const codeSpans = []
  let rendered = value.replace(/`([^`]+)`/g, (_, code) => {
    codeSpans.push(`<code>${escapeHtml(code)}</code>`)
    return `@@CODE${codeSpans.length - 1}@@`
  })

  rendered = escapeHtml(rendered)
  rendered = rendered.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, href) => {
    return `<a href="${escapeHtml(href)}">${escapeHtml(label)}</a>`
  })
  rendered = rendered.replace(/@@CODE(\d+)@@/g, (_, index) => codeSpans[Number(index)])

  return rendered
}

function renderMarkdown(source) {
  const lines = source.split('\n')
  const blocks = []
  let index = 0

  while (index < lines.length) {
    const line = lines[index]

    if (!line.trim()) {
      index += 1
      continue
    }

    if (line.startsWith('```')) {
      const language = line.slice(3).trim()
      const code = []
      index += 1

      while (index < lines.length && !lines[index].startsWith('```')) {
        code.push(lines[index])
        index += 1
      }

      index += 1
      const languageClass = language ? ` class="language-${escapeHtml(language)}"` : ''
      blocks.push(`<pre><code${languageClass}>${escapeHtml(code.join('\n'))}</code></pre>`)
      continue
    }

    const heading = line.match(/^(#{2,6})\s+(.+)$/)
    if (heading) {
      const level = heading[1].length
      blocks.push(`<h${level}>${renderInline(heading[2])}</h${level}>`)
      index += 1
      continue
    }

    if (/^-\s+/.test(line)) {
      const items = []

      while (index < lines.length && /^-\s+/.test(lines[index])) {
        items.push(`<li>${renderInline(lines[index].replace(/^-\s+/, ''))}</li>`)
        index += 1
      }

      blocks.push(`<ul>${items.join('')}</ul>`)
      continue
    }

    const paragraph = []
    while (
      index < lines.length &&
      lines[index].trim() &&
      !lines[index].startsWith('```') &&
      !/^(#{2,6})\s+/.test(lines[index]) &&
      !/^-\s+/.test(lines[index])
    ) {
      paragraph.push(lines[index])
      index += 1
    }

    blocks.push(`<p>${renderInline(paragraph.join(' '))}</p>`)
  }

  return blocks.join('\n')
}

function compareTagsDescending(left, right) {
  const leftParts = left.replace(/^v/, '').split(/[.-]/)
  const rightParts = right.replace(/^v/, '').split(/[.-]/)
  const maxLength = Math.max(leftParts.length, rightParts.length)

  for (let index = 0; index < maxLength; index += 1) {
    const leftPart = leftParts[index] ?? '0'
    const rightPart = rightParts[index] ?? '0'
    const leftNumber = Number(leftPart)
    const rightNumber = Number(rightPart)

    if (Number.isInteger(leftNumber) && Number.isInteger(rightNumber) && leftNumber !== rightNumber) {
      return rightNumber - leftNumber
    }

    if (leftPart !== rightPart) {
      return rightPart.localeCompare(leftPart)
    }
  }

  return right.localeCompare(left)
}

const releases = Object.entries(releaseFiles)
  .map(([file, source]) => {
    const tag = extractTag(file)
    const body = extractReleaseBody(source, file)

    return {
      tag,
      title: extractTitle(source, tag),
      html: renderMarkdown(body)
    }
  })
  .sort((left, right) => compareTagsDescending(left.tag, right.tag))
</script>

<template>
  <section v-for="release in releases" :key="release.tag" class="vy-release-entry">
    <h2 :id="release.tag">{{ release.title }}</h2>
    <div v-html="release.html"></div>
  </section>
</template>
