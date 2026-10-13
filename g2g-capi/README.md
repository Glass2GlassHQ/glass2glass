# g2g-capi

C ABI for [glass2glass](https://github.com/boxerab/glass2glass): parse a launch
line, run the pipeline, pump the bus, and push or pull frames through
appsrc/appsink from any language. Ships as a cdylib / staticlib plus `g2g.h`.

## Changing a running pipeline

A launched pipeline accepts edits while it runs. Elements are addressed by their
instance name: the `name=` from the launch line, or the name an insert or
replace returned. A new element is one launch-line element such as
`"valve drop=true"`, without `name=` or `log-category=`.

```c
char *err = NULL;
G2gPipeline *p = g2g_pipeline_launch(
    "videotestsrc name=src ! clocksync ! fakesink name=sink", &err);
char *gate = g2g_pipeline_insert_after(p, "src", "valve drop=false", &err);
g2g_pipeline_set_property(p, gate, "drop", "true", &err);
char *drop = g2g_pipeline_get_property(p, gate, "drop", &err);   /* "true" */
g2g_string_free(drop);
g2g_pipeline_remove(p, gate, &err);
g2g_string_free(gate);
```

`g2g_pipeline_insert_after` and `g2g_pipeline_insert_before` splice the element
onto the edge below or above a node. `g2g_pipeline_remove` lifts a transform out
and drops it. `g2g_pipeline_replace_sink` and `g2g_pipeline_replace_source` swap
an end of the pipeline for a new element. `g2g_pipeline_set_property` takes the
value as launch-line text and `g2g_pipeline_get_property` returns it the same
way, or NULL with no error when the element has no such property.

A refused edit returns NULL or 0, writes the reason (`UnknownNode`,
`NotMutable`, `PropertyRejected`, ...) to `err_out`, and leaves the pipeline
running as it was. Free every returned string and error with `g2g_string_free`.
Each call blocks until the element upstream of the edit pushes its next packet,
so an edit on a stalled pipeline waits for it.
