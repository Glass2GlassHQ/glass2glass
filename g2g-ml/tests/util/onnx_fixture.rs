// Shared hand-encoded ONNX fixture builder (included via `include!` by the
// integration tests; not a test crate itself, so plain comments only).
// Wire format only needs varints and length-delimited fields. Field numbers
// follow onnx.proto3 (ModelProto / GraphProto / NodeProto / ValueInfoProto).

pub(crate) fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

pub(crate) fn field_varint(field: u64, v: u64, out: &mut Vec<u8>) {
    varint(field << 3, out); // wire type 0
    varint(v, out);
}

pub(crate) fn field_bytes(field: u64, bytes: &[u8], out: &mut Vec<u8>) {
    varint((field << 3) | 2, out); // wire type 2
    varint(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
}

pub(crate) fn field_str(field: u64, s: &str, out: &mut Vec<u8>) {
    field_bytes(field, s.as_bytes(), out);
}

/// ValueInfoProto: name=1, type=2; TypeProto.tensor_type=1 holds
/// elem_type=1 (FLOAT = 1) and shape=2 (dims as Dimension.dim_value=1).
pub(crate) fn value_info(name: &str, dims: &[u64]) -> Vec<u8> {
    let mut shape = Vec::new();
    for d in dims {
        let mut dim = Vec::new();
        field_varint(1, *d, &mut dim);
        field_bytes(1, &dim, &mut shape);
    }
    let mut tensor_type = Vec::new();
    field_varint(1, 1, &mut tensor_type); // elem_type = FLOAT
    field_bytes(2, &shape, &mut tensor_type);
    let mut type_proto = Vec::new();
    field_bytes(1, &tensor_type, &mut type_proto);
    let mut vi = Vec::new();
    field_str(1, name, &mut vi);
    field_bytes(2, &type_proto, &mut vi);
    vi
}

/// A complete ModelProto holding one `Identity` node from input "x" to
/// output "y", both f32 tensors of `dims` (so a test can also build a
/// contract-violating model, e.g. 4 channels).
pub(crate) fn identity_model(dims: &[u64]) -> Vec<u8> {
    let mut node = Vec::new();
    field_str(1, "x", &mut node); // input
    field_str(2, "y", &mut node); // output
    field_str(4, "Identity", &mut node); // op_type

    let mut graph = Vec::new();
    field_bytes(1, &node, &mut graph);
    field_str(2, "g", &mut graph); // graph name
    field_bytes(11, &value_info("x", dims), &mut graph); // input
    field_bytes(12, &value_info("y", dims), &mut graph); // output

    let mut opset = Vec::new();
    field_str(1, "", &mut opset); // default domain
    field_varint(2, 13, &mut opset); // opset 13

    let mut model = Vec::new();
    field_varint(1, 8, &mut model); // ir_version = 8
    field_bytes(7, &graph, &mut model);
    field_bytes(8, &opset, &mut model);
    model
}
