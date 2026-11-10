_g2g_options() {
    "$1" --help 2>/dev/null | sed '/^$/q' | grep -oE '(^|[[ ])-{1,2}[a-z][a-z-]*' | tr -d '[ ' | sort -u
}

_g2g_plugin_arguments() {
    local i
    for ((i = 1; i < COMP_CWORD; i++)); do
        if [[ ${COMP_WORDS[i]} == --plugin ]]; then
            printf '%s\n' --plugin "${COMP_WORDS[i + 1]}"
        fi
    done
}

_g2g_element_names() {
    local -a plugins
    mapfile -t plugins < <(_g2g_plugin_arguments)
    g2g-inspect "${plugins[@]}" 2>/dev/null | cut -d: -f1
}

_g2g_property_names() {
    local -a plugins
    mapfile -t plugins < <(_g2g_plugin_arguments)
    g2g-inspect "${plugins[@]}" "$1" 2>/dev/null |
        awk '/^Element Properties:/ { listing = 1; next } listing && /^  [a-z0-9_-]+:/ { sub(/^  /, ""); sub(/:.*/, ""); print }'
}

_g2g_inspect() {
    local cur=${COMP_WORDS[COMP_CWORD]} prev=${COMP_WORDS[COMP_CWORD - 1]}
    case $prev in
        # complete -o default lists file names for an empty reply
        --plugin | --trusted-key | --gst-scan) COMPREPLY=(); return ;;
        # a reply of the word itself completes nothing
        --gst) COMPREPLY=("$cur"); return ;;
    esac
    if [[ $cur == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "$(_g2g_options g2g-inspect)" -- "$cur")
        return
    fi
    mapfile -t COMPREPLY < <(compgen -W "$(_g2g_element_names)" -- "$cur")
}

_g2g_launch() {
    local cur=${COMP_WORDS[COMP_CWORD]} prev=${COMP_WORDS[COMP_CWORD - 1]}
    case $prev in
        --plugin | --graph | --script | --record-on-error | --trace | --profile | --placement-profile | =)
            COMPREPLY=(); return ;;
        --objective)
            mapfile -t COMPREPLY < <(compgen -W 'latency throughput' -- "$cur"); return ;;
        --observe | --observe-host | --seek | --seek-at | :)
            COMPREPLY=("$cur"); return ;;
    esac

    # bash splits `key=value` and `position:seconds` into three words
    local element= expecting_element=1 skip_next=0 word i
    for ((i = 1; i < COMP_CWORD; i++)); do
        word=${COMP_WORDS[i]}
        if ((skip_next)); then
            skip_next=0
            continue
        fi
        case $word in
            = | :) skip_next=1; continue ;;
            '!') expecting_element=1; continue ;;
            --plugin | --graph | --script | --record-on-error | --trace | --profile | --placement-profile | --objective | --observe | --observe-host | --seek | --seek-at)
                skip_next=1; continue ;;
            -*) continue ;;
        esac
        if ((expecting_element)); then
            element=$word
            expecting_element=0
        fi
    done

    if [[ -z $element && $cur == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "$(_g2g_options g2g-launch)" -- "$cur")
        return
    fi
    if ((expecting_element)); then
        mapfile -t COMPREPLY < <(compgen -W "$(_g2g_element_names)" -- "$cur")
        return
    fi
    # caps strings and `name.` pad references have no properties
    if [[ $element == */* || $element == *. ]]; then
        COMPREPLY=('!')
        return
    fi
    mapfile -t COMPREPLY < <(compgen -W "$(_g2g_property_names "$element" | sed 's/$/=/') !" -- "$cur")
    if [[ ${#COMPREPLY[@]} -eq 1 && ${COMPREPLY[0]} == *= ]]; then
        compopt -o nospace
    fi
}

complete -o default -F _g2g_inspect g2g-inspect
complete -o default -F _g2g_launch g2g-launch
