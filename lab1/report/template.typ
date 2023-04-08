// The project function defines how your document looks.
// It takes your content and some metadata and formats it.
// Go ahead and customize it to your liking!
#let project(title: "", authors: (), body) = {
  // Set the document's basic properties.
  set document(author: authors, title: title)
  set page(numbering: "1", number-align: center)
 // set text(font: "Linux Libertine", lang: "en")

  // Title row.
  align(center)[
    #block(text(weight: 700, 1.75em, title))
  ]

  // Author information.
  pad(
    top: 0.5em,
    bottom: 0.5em,
    x: 2em,
    grid(
      columns: (1fr,) * calc.min(3, authors.len()),
      gutter: 1em,
      ..authors.map(author => align(center, strong(author))),
    ),
  )

  // Main body.
  set text(kerning: true)
  set par(justify: true, linebreaks: "optimized")
  set text(font: "Source Han Serif SC", lang: "zh")
  show math.equation: set text(font: "New Computer Modern Math", weight: 400)
  show link: underline
  body
}

#let base_env(type: "Theorem", name: none, numbered: true, fg: black, bg: white, body) = locate(
    location => {
      let lvl = counter(heading).at(location)
      let i = counter(type).at(location).first()
      let top = if lvl.len() > 0 { lvl.first() } else { 0 }
      show: block. with(spacing: 11.5pt)
      stack(
        dir: ttb,
        rect(fill: fg, radius : (top-right: 5pt), stroke: fg)[
          #strong(
            text(white)[
              #type
              #if name != none [ (#name) ]
            ]
          )
        ],
        rect(
          width : 100%,
          fill: bg,
          inset: 1em,
          radius: ( right: 5pt ),
          stroke: (
            left: fg,
            right: bg + 0pt,
            top: bg + 0pt,
            bottom: bg + 0pt,
          )
        )[
          #body
        ]
      )
    }
  )
)

#let info(body, name: none, numbered: true) = {
  base_env(
    type: "说明",
    name: name,
    numbered: numbered,
    fg: blue,
    bg: rgb( "#e8e8f8"),
    body
  )
}

#let chinese = true

#let normal-font = if chinese {
  ("Times New Roman"," FandolSong")
} else {
  "New Computer Modern"
}