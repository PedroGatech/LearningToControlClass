using Documenter
# using Pluto

include("definitions.jl")

repo_dir = dirname(@__DIR__)
build_dir = joinpath(repo_dir, "docs", "build")

plutos = [
    joinpath(repo_dir, "class01", "background_materials", "math_basics.html"),
    joinpath(repo_dir, "class01", "background_materials", "optimization_basics.html"),
    joinpath(repo_dir, "class01", "background_materials", "optimization_motivation.html"),
    joinpath(repo_dir, "class01", "class01_intro.html"),
    joinpath(repo_dir, "class02", "part1_minimization.html"),
    joinpath(repo_dir, "class02", "part1_root_finding.html"),
    joinpath(repo_dir, "class02", "part2_eq_constraints.html"),
    joinpath(repo_dir, "class02", "part3_ipm.html"),
    joinpath(repo_dir, "class05", "class05.html"),
]

if !isdir(build_dir)
    symlink(joinpath(repo_dir, "class01"),
        joinpath(repo_dir, "docs", "src", "class01")
    )
    symlink(joinpath(repo_dir, "class02"),
        joinpath(repo_dir, "docs", "src", "class02")
    )
    symlink(joinpath(repo_dir, "class05"),
        joinpath(repo_dir, "docs", "src", "class05")
    )
end

makedocs(
    sitename = "LearningToControlClass",
    format = Documenter.HTML(;
        assets = ["assets/wider.css", "assets/redlinks.css"],
        mathengine = Documenter.MathJax3(Dict(
            :tex => Dict(
                "macros" => make_macros_dict("docs/src/assets/definitions.tex"),
                "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
                "tags" => "ams",
            ),
        )),
    ),
    pages  = [
        "Home"   => "index.md",
        "Class 1" => ["class01/class01.md",
            "class01/background_materials/git_adventure_guide.md",
        ],
        "Class 2" => "class02/overview.md",
        "Class 5" => "class05/class05.md",
    ],
)

for pluto in plutos
    filename = replace(pluto, repo_dir => build_dir)
    mkpath(dirname(filename))
    cp(pluto, filename, force=true)
end

rm(joinpath(repo_dir, "docs", "src", "class01"), force=true)
rm(joinpath(repo_dir, "docs", "src", "class02"), force=true)
rm(joinpath(repo_dir, "docs", "src", "class05"), force=true)

# In case we want to generate HTML from Pluto notebooks in CI
# plutos = [
#     joinpath(repo_dir, "class01", "background_materials", "math_basics.jl"),
#     joinpath(repo_dir, "class01", "background_materials", "optimization_basics.jl"),
#     joinpath(repo_dir, "class01", "background_materials", "optimization_motivation.jl"),
#     joinpath(repo_dir, "class01", "class01_intro.jl"),
# ]
# s = Pluto.ServerSession();
# for pluto in plutos
#     nb = Pluto.SessionActions.open(s, pluto; run_async=false)
#     html_contents = Pluto.generate_html(nb; binder_url_js="undefined")
#     filename = replace(pluto, repo_dir => build_dir)
#     html_path = replace(filename, r"\.jl$" => ".html")
#     mkpath(dirname(html_path))
#     open(html_path, "w") do f
#         write(f, html_contents)
#     end
# end

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/LearningToOptimize/LearningToControlClass.git",
    push_preview=true,
)
