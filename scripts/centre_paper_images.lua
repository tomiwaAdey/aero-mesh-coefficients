-- Centre standalone manuscript images without turning their alt text into captions.
function Para(paragraph)
  if #paragraph.content == 1 and paragraph.content[1].tag == "Image" then
    return {
      pandoc.RawBlock("latex", "\\begin{center}"),
      paragraph,
      pandoc.RawBlock("latex", "\\end{center}"),
    }
  end
end
