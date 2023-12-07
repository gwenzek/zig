const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

test "@callerSrc" {
    // DO NOT MOVE, the test case is dependent on the position of this call.
    try callerSrcDoTheTest();
}

fn callerSrcDoTheTest() !void {
    const caller_src = @callerSrc();

    try testing.expect(std.mem.endsWith(u8, caller_src.file, "caller_src.zig"));
    try testing.expectEqualStrings("test.@callerSrc", caller_src.fn_name);

    // TODO: read proper line info
    try testing.expectEqual(caller_src.line, 0);
    try testing.expectEqual(caller_src.column, 0);

    // Check we use null terminated strings.
    try testing.expectEqual(caller_src.fn_name[caller_src.fn_name.len], 0);
    try testing.expectEqual(caller_src.file[caller_src.file.len], 0);
}

test "@callerSrc2" {
    // Note: the result of @callerSrc aren't updated.
    try callerSrcDoTheTest();
}
