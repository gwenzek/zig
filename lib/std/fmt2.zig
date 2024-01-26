const std = @import("std");
const builtin = @import("builtin");

const io = std.io;
const math = std.math;
const assert = std.debug.assert;
const mem = std.mem;
const unicode = std.unicode;
const meta = std.meta;

const lossyCast = std.math.lossyCast;
const FormatOptions = std.fmt.FormatOptions;

pub const default_max_depth = 3;
pub const max_format_args = 32;
pub const FmtFn = *const fn (bytes: []const u8, buf: []u8) anyerror!usize;
pub const WriteFn = *const fn (context: *const anyopaque, bytes: []const u8) anyerror!usize;

pub fn fmtFn(comptime T: type) FmtFn {
    const A = struct {
        pub fn fmt(bytes: []const u8, buf: []u8) anyerror!usize {
            // TODO: .fmt() should accept a runtime write function instead of comptime.
            // That way we could also pass write function at runtime and won't require the tmp buf.
            var fbs = std.io.fixedBufferStream(buf);
            const value: *const T = @alignCast(@ptrCast(bytes.ptr));
            try value.format("", .{}, fbs.writer());
            return fbs.getWritten().len;
        }
    };
    return A.fmt;
}

pub fn writeFn(comptime Writer: type) WriteFn {
    const A = struct {
        pub fn write(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const ctx: *const Writer = @alignCast(@ptrCast(context));
            return ctx.write(bytes) catch return error.NoSpaceLeft;
        }
    };
    return A.write;
}

pub fn ReadOptional(comptime T_opt: type) type {
    return struct {
        pub fn read(opq: []const u8) ?[]const u8 {
            const x: *const T_opt = @alignCast(@ptrCast(opq.ptr));
            if (x.*) |y| {
                const T_inner = @TypeOf(y);
                // This assumes the optional tag is stored **AFTER** the payload.
                return opq[0..@sizeOf(T_inner)];
            } else {
                return null;
            }
        }
    };
}

const ErrorOrPayloadEnum = enum(u2) {
    err,
    payload,
};

const ErrorOrPayload = union(ErrorOrPayloadEnum) {
    err: []const u8,
    payload: []const u8,
};

pub fn ReadErrorUnion(comptime T_payload: type) type {
    return struct {
        pub fn read(opq: []const u8) ErrorOrPayload {
            const x: *const anyerror!T_payload = @alignCast(@ptrCast(opq.ptr));
            _ = x.* catch |err| {
                return .{ .err = @errorName(err) };
            };
            // This assumes the error tag is stored **AFTER** the payload.
            return .{ .payload = opq[0..@sizeOf(T_payload)] };
        }
    };
}

pub const FmtTypeEnum = enum {
    boolean,
    integer,
    float,
    literal,
    substr,
    arr,
    buff,
    any,
    optional,
    error_union,
    err,
    not_implemented,
};

pub const SmallSlice = struct { start: u16, end: u16 };

pub const FmtType = union(FmtTypeEnum) {
    boolean: struct { offset: u16, value: ?bool = null },
    integer: struct { bytes: SmallSlice, base: u8, signedness: std.builtin.Signedness },
    float: u0,
    literal: []const u8,
    substr: SmallSlice,
    arr: SmallSlice,
    buff: u0,
    // output a value of any type using its default format.
    any: FmtFn,
    optional: struct {
        bytes: SmallSlice,
        skip: u8,
        read: *const fn (opq: []const u8) ?[]const u8,
    },
    error_union: struct {
        bytes: SmallSlice,
        skip: u8,
        read: *const fn (opq: []const u8) ErrorOrPayload,
    },
    err: u32,
    not_implemented: u0,
};

pub const FmtToken = struct {
    type: FmtType,
    options: FormatOptions,
    // comptime_data: bool,

    wrapping: enum {
        none,
        // output optional value as either the unwrapped value, or `null`; may be followed by a format specifier for the underlying value.
        optional,
        // output error union value as either the unwrapped value, or the formatted error value; may be followed by a format specifier for the underlying value.
        err,
        // output the address of the value instead of the value itself.
        addr,
    } = .none,
};

pub const FmtState = struct {
    fmt: []const u8,
    tokens: [2 * max_format_args + 1]FmtToken = undefined,
    num_tokens: u6 = 0,

    pub fn append(self: *FmtState, tok: FmtType, options: FormatOptions) void {
        self.tokens[self.num_tokens].type = tok;
        self.tokens[self.num_tokens].options = options;
        self.num_tokens += 1;
    }

    pub fn write(self: FmtState, write_ctx: *const anyopaque, write_fn: WriteFn, value: []const u8) !void {
        var i: usize = 0;
        var j: usize = i;
        while (i < self.num_tokens) {
            i = try self.writeToken(write_ctx, write_fn, value, i);
            std.debug.assert(i > j);
            j = i;
        }
    }

    fn writeToken(self: FmtState, ctx: *const anyopaque, write_fn: WriteFn, value: []const u8, i: usize) !usize {
        var buf: [16]u8 = undefined;
        const token = self.tokens[i];
        _ = switch (token.type) {
            .boolean => |b| {
                const val = b.value orelse (value[b.offset] != 0);
                _ = try write_fn(ctx, if (val) "true" else "false");
            },
            .integer => try write_fn(ctx, fmtInt(&buf, value, token)),
            .literal => |lit| try write_fn(ctx, lit),
            .substr => |sub| try write_fn(ctx, self.fmt[sub.start..sub.end]),
            // TODO: handle left/right/center alignment
            // TODO: comptime dereference requires '[]const u8' to have a well-defined layout, but it does not.
            .buff => try write_fn(ctx, @as(*const []const u8, @alignCast(@ptrCast(value))).*),
            .arr => |arr| try write_fn(ctx, value[arr.start..arr.end]),
            .optional => |opt| {
                if (opt.read(value)) |bytes| {
                    return self.writeToken(ctx, write_fn, bytes, i + 1);
                } else {
                    _ = try write_fn(ctx, "null");
                    return i + opt.skip + 1;
                }
            },
            .error_union => |error_union| {
                switch (error_union.read(value)) {
                    .err => |err_name| {
                        _ = try write_fn(ctx, err_name);
                        return i + error_union.skip + 1;
                    },
                    .payload => |bytes| {
                        return self.writeToken(ctx, write_fn, bytes, i + 1);
                    },
                }
            },
            .any => |fmt_fn| {
                var big_buf: [1024]u8 = undefined;
                const n = fmt_fn(value, &big_buf) catch n: {
                    big_buf[1021..1024].* = "...".*;
                    break :n big_buf.len;
                };
                _ = try write_fn(ctx, big_buf[0..n]);
            },
            else => {
                // std.log.warn("fmt doesn't support {} yet", .{token.type});
            },
        };
        return i + 1;
    }

    pub fn count(self: FmtState, value: []const u8) u64 {
        var counting_writer = std.io.countingWriter(std.io.null_writer);
        self.write(counting_writer.writer(), value) catch |err| switch (err) {};
        return counting_writer.bytes_written;
    }
};

pub fn fmtInt(buf: []u8, value: []const u8, token: FmtToken) []const u8 {
    const bytes = token.type.integer.bytes;
    const size = bytes.end - bytes.start;
    const data: [*]const u8 = value[bytes.start..bytes.end].ptr;
    switch (token.type.integer.signedness) {
        .unsigned => {
            const val: u64 = switch (size) {
                1 => @as(*const u8, @alignCast(@ptrCast(data))).*,
                2 => @as(*const u16, @alignCast(@ptrCast(data))).*,
                4 => @as(*const u32, @alignCast(@ptrCast(data))).*,
                8 => @as(*const u64, @alignCast(@ptrCast(data))).*,
                else => unreachable,
            };
            return formatInt(buf, val, token.type.integer.base, .lower, token.options);
        },
        .signed => {
            const val: i64 = switch (size) {
                1 => @as(*const i8, @alignCast(@ptrCast(data))).*,
                2 => @as(*const i16, @alignCast(@ptrCast(data))).*,
                4 => @as(*const i32, @alignCast(@ptrCast(data))).*,
                8 => @as(*const i64, @alignCast(@ptrCast(data))).*,
                else => unreachable,
            };
            return formatInt(buf, val, token.type.integer.base, .lower, token.options);
        },
    }
}

pub fn formatInt(
    buf: []u8,
    value: anytype,
    base: u8,
    case: std.fmt.Case,
    options: FormatOptions,
) []u8 {
    assert(base >= 2);

    const int_value = if (@TypeOf(value) == comptime_int) blk: {
        const Int = math.IntFittingRange(value, value);
        break :blk @as(Int, value);
    } else value;

    const value_info = @typeInfo(@TypeOf(int_value)).Int;

    // The type must have the same size as `base` or be wider in order for the
    // division to work
    const min_int_bits = comptime @max(value_info.bits, 8);
    const MinInt = std.meta.Int(.unsigned, min_int_bits);

    const abs_value = @abs(int_value);

    var a: MinInt = abs_value;
    var index: usize = buf.len;

    if (base == 10) {
        while (a >= 100) : (a = @divTrunc(a, 100)) {
            index -= 2;
            buf[index..][0..2].* = digits2(@as(usize, @intCast(a % 100)));
        }

        if (a < 10) {
            index -= 1;
            buf[index] = '0' + @as(u8, @intCast(a));
        } else {
            index -= 2;
            buf[index..][0..2].* = digits2(@as(usize, @intCast(a)));
        }
    } else {
        while (true) {
            const digit = a % base;
            index -= 1;
            buf[index] = std.fmt.digitToChar(@as(u8, @intCast(digit)), case);
            a /= base;
            if (a == 0) break;
        }
    }

    if (value_info.signedness == .signed) {
        if (value < 0) {
            // Negative integer
            index -= 1;
            buf[index] = '-';
        } else if (options.width == null or options.width.? == 0) {
            // Positive integer, omit the plus sign
        } else {
            // Positive integer
            index -= 1;
            buf[index] = '+';
        }
    }

    return buf[index..];
}

fn digits2(value: usize) [2]u8 {
    return ("0001020304050607080910111213141516171819" ++
        "2021222324252627282930313233343536373839" ++
        "4041424344454647484950515253545556575859" ++
        "6061626364656667686970717273747576777879" ++
        "8081828384858687888990919293949596979899")[value * 2 ..][0..2].*;
}

/// Parses a format string.
///
/// The format string must be comptime-known and may contain placeholders following
/// this format:
/// `{[argument][specifier]:[fill][alignment][width].[precision]}`
///
/// Above, each word including its surrounding [ and ] is a parameter which you have to replace with something:
///
/// - *argument* is either the numeric index or the field name of the argument that should be inserted
///   - when using a field name, you are required to enclose the field name (an identifier) in square
///     brackets, e.g. {[score]...} as opposed to the numeric index form which can be written e.g. {2...}
/// - *specifier* is a type-dependent formatting option that determines how a type should formatted (see below)
/// - *fill* is a single character which is used to pad the formatted text
/// - *alignment* is one of the three characters `<`, `^`, or `>` to make the text left-, center-, or right-aligned, respectively
/// - *width* is the total width of the field in characters
/// - *precision* specifies how many decimals a formatted number should have
///
/// Note that most of the parameters are optional and may be omitted. Also you can leave out separators like `:` and `.` when
/// all parameters after the separator are omitted.
/// Only exception is the *fill* parameter. If *fill* is required, one has to specify *alignment* as well, as otherwise
/// the digits after `:` is interpreted as *width*, not *fill*.
///
/// The *specifier* has several options for types:
/// - `x` and `X`: output numeric value in hexadecimal notation
/// - `s`:
///   - for pointer-to-many and C pointers of u8, print as a C-string using zero-termination
///   - for slices of u8, print the entire slice as a string without zero-termination
/// - `e`: output floating point value in scientific notation
/// - `d`: output numeric value in decimal notation
/// - `b`: output integer value in binary notation
/// - `o`: output integer value in octal notation
/// - `c`: output integer as an ASCII character. Integer type must have 8 bits at max.
/// - `u`: output integer as an UTF-8 sequence. Integer type must have 21 bits at max.
/// - `?`: output optional value as either the unwrapped value, or `null`; may be followed by a format specifier for the underlying value.
/// - `!`: output error union value as either the unwrapped value, or the formatted error value; may be followed by a format specifier for the underlying value.
/// - `*`: output the address of the value instead of the value itself.
/// - `any`: output a value of any type using its default format.
///
/// If a formatted user type contains a function of the type
/// ```
/// pub fn format(value: ?, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void
/// ```
/// with `?` being the type formatted, this function will be called instead of the default implementation.
/// This allows user types to be formatted in a logical manner instead of dumping all fields of the type.
///
/// A user type may be a `struct`, `vector`, `union` or `enum` type.
///
/// To print literal curly braces, escape them by writing them twice, e.g. `{{` or `}}`.
pub fn format(
    writer: anytype,
    comptime fmt: []const u8,
    args: anytype,
) @TypeOf(writer).Error!void {
    const ArgsType = @TypeOf(args);
    var formatter = comptime parseFormat(fmt, ArgsType);
    _ = &args;
    const args_bytes: []const u8 = @as([*]const u8, @ptrCast(&args))[0..@sizeOf(ArgsType)];

    const Writer = @TypeOf(writer);
    formatter.write(&writer, writeFn(Writer), args_bytes) catch |err| {
        const maybe_error_set = @typeInfo(Writer.Error).ErrorSet;
        if (maybe_error_set) |error_set| {
            if (error_set.len == 0) return;
            return @as(Writer.Error, @errorCast(err));
        }
        // unreachable
        return;
    };
}

const CountingWriter = @TypeOf(std.io.countingWriter(std.io.null_writer));

pub fn parseFormat(
    comptime fmt: []const u8,
    comptime ArgsType: type,
) FmtState {
    const args_type_info = @typeInfo(ArgsType);
    if (args_type_info != .Struct) {
        @compileError("expected tuple or struct argument, found " ++ @typeName(ArgsType));
    }

    const fields_info = args_type_info.Struct.fields;
    if (fields_info.len > max_format_args) {
        @compileError("32 arguments max are supported per format call");
    }
    if (fmt.len >= std.math.maxInt(u16)) {
        @compileError("format string is too long.");
    }

    comptime var res: FmtState = .{ .fmt = fmt };
    @setEvalBranchQuota(2000000);
    comptime var arg_state: std.fmt.ArgState = .{ .args_len = fields_info.len };
    comptime var i = 0;
    inline while (i < fmt.len) {
        const start_index = i;

        inline while (i < fmt.len) : (i += 1) {
            switch (fmt[i]) {
                '{', '}' => break,
                else => {},
            }
        }

        comptime var end_index = i;
        comptime var unescape_brace = false;

        // Handle {{ and }}, those are un-escaped as single braces
        if (i + 1 < fmt.len and fmt[i + 1] == fmt[i]) {
            unescape_brace = true;
            // Make the first brace part of the literal...
            end_index += 1;
            // ...and skip both
            i += 2;
        }

        // Write out the literal
        if (start_index != end_index) {
            comptime {
                res.append(.{ .substr = .{ .start = start_index, .end = end_index } }, .{});
            }
        }

        // We've already skipped the other brace, restart the loop
        if (unescape_brace) continue;

        if (i >= fmt.len) break;

        if (fmt[i] == '}') {
            @compileError("missing opening {");
        }

        // Get past the {
        comptime assert(fmt[i] == '{');
        i += 1;

        const fmt_begin = i;
        // Find the closing brace
        inline while (i < fmt.len and fmt[i] != '}') : (i += 1) {}
        const fmt_end = i;

        if (i >= fmt.len) {
            @compileError("missing closing }");
        }

        // Get past the }
        comptime assert(fmt[i] == '}');
        i += 1;

        const placeholder = comptime std.fmt.Placeholder.parse(fmt[fmt_begin..fmt_end].*);
        const arg_pos = comptime switch (placeholder.arg) {
            .none => null,
            .number => |pos| pos,
            .named => |arg_name| meta.fieldIndex(ArgsType, arg_name) orelse
                @compileError("no argument with name '" ++ arg_name ++ "'"),
        };

        const width = switch (placeholder.width) {
            .none => null,
            .number => |v| v,
            .named => |arg_name| blk: {
                const arg_i = comptime meta.fieldIndex(ArgsType, arg_name) orelse
                    @compileError("no argument with name '" ++ arg_name ++ "'");
                _ = comptime arg_state.nextArg(arg_i) orelse @compileError("too few arguments");
                // TODO: try reading the actual value
                // const value = args_type_info.Struct.fields[arg_i].default_value.?.*;
                // const width: usize = @bitCast(value);
                break :blk null;
            },
        };

        const precision = switch (placeholder.precision) {
            .none => null,
            .number => |v| v,
            // TODO named argument not supported
            .named => |arg_name| blk: {
                const arg_i = comptime meta.fieldIndex(ArgsType, arg_name) orelse
                    @compileError("no argument with name '" ++ arg_name ++ "'");
                _ = comptime arg_state.nextArg(arg_i) orelse @compileError("too few arguments");
                // TODO: try reading the actual value
                // const value = args_type_info.Struct.fields[arg_i].default_value.?.*;
                // const precision: usize = @bitCast(value);
                break :blk null;
            },
        };

        const arg_to_print = comptime arg_state.nextArg(arg_pos) orelse
            @compileError("too few arguments");

        comptime {
            const field = args_type_info.Struct.fields[arg_to_print];
            const comptime_data: ?[]const u8 = if (field.is_comptime) comptimeData(field) else null;
            addTokensForType(
                field.type,
                placeholder.specifier_arg,
                &res,
                comptime_data,
                FormatOptions{
                    .fill = placeholder.fill,
                    .alignment = placeholder.alignment,
                    .width = width,
                    .precision = precision,
                },
                std.options.fmt_max_depth,
                if (field.is_comptime) 0 else @offsetOf(ArgsType, field.name),
            );
        }
    }

    if (comptime arg_state.hasUnusedArgs()) {
        const missing_count = arg_state.args_len - @popCount(arg_state.used_args);
        switch (missing_count) {
            0 => unreachable,
            1 => @compileError("unused argument in '" ++ fmt ++ "'"),
            else => @compileError(digits2(missing_count) ++ " unused arguments in '" ++ fmt ++ "'"),
        }
    }

    return res;
}

/// Returns a runtime readable memory slice for a given comptime data.
/// This may require a copy for int and floats.
fn comptimeData(comptime field: std.builtin.Type.StructField) ?[]const u8 {
    const T = field.type;
    const value_ptr: *const T = @alignCast(@ptrCast(field.default_value.?));
    const value: T = value_ptr.*;
    comptime {
        switch (@typeInfo(T)) {
            .Null => {
                return null;
            },
            .Pointer => |ptr_info| switch (ptr_info.size) {
                .One => {
                    const ptr: [*]const u8 = @ptrCast(value);
                    var len = @sizeOf(ptr_info.child);
                    // Strip sentinel
                    switch (@typeInfo(ptr_info.child)) {
                        .Array => |array_info| {
                            if (array_info.sentinel) |_| {
                                len -= @sizeOf(array_info.child);
                            }
                        },
                        else => {},
                    }
                    return ptr[0..len];
                },
                .Many, .C => {
                    return std.mem.span(value);
                },
                .Slice => {
                    return value;
                },
            },
            .ComptimeInt => {
                const Int = if (value > 0) u64 else i64;
                const buf: [8]u8 = @bitCast(@as(Int, @intCast(value)));
                return &buf;
            },
            .ComptimeFloat => {
                const buf: [8]u8 = @bitCast(@as(f64, @floatCast(value)));
                return &buf;
            },
            .Type => {
                // @compileLog("compile time type: ", value, @typeName(value));
                return @typeName(value);
            },
            else => {
                const buf: [*]const u8 = @ptrCast(value_ptr);
                // @compileLog("compile time val: ", value_ptr, buf);
                return buf[0..@sizeOf(T)];
            },
        }
    }
}

// This ANY const is a workaround for: https://github.com/ziglang/zig/issues/7948
const ANY = "any";

fn stripOptionalOrErrorUnionSpec(comptime fmt: []const u8) []const u8 {
    return if (std.mem.eql(u8, fmt[1..], ANY))
        ANY
    else
        fmt[1..];
}

pub fn invalidFmtError(comptime fmt: []const u8, comptime T: type) void {
    @compileError("invalid format string '" ++ fmt ++ "' for type '" ++ @typeName(T) ++ "'");
}

pub fn addTokensForType(
    comptime T: type,
    comptime fmt: []const u8,
    state: *FmtState,
    comptime_data: ?[]const u8,
    options: FormatOptions,
    max_depth: usize,
    base_offset: u16,
) void {
    const actual_fmt = comptime if (std.mem.eql(u8, fmt, ANY))
        std.fmt.defaultSpec(T)
    else if (fmt.len != 0 and (fmt[0] == '?' or fmt[0] == '!')) switch (@typeInfo(T)) {
        .Optional, .ErrorUnion => fmt,
        else => stripOptionalOrErrorUnionSpec(fmt),
    } else fmt;

    if (comptime std.mem.eql(u8, actual_fmt, "*")) {
        addAddressTokens(state, T);
        return;
    }

    if (std.meta.hasFn(T, "format")) {
        state.append(.{ .any = fmtFn(T) }, options);
        return;
    }

    switch (@typeInfo(T)) {
        .ComptimeInt => {
            comptime var buf: [64]u8 = undefined;
            const value: *const i64 = @alignCast(@ptrCast(comptime_data.?));
            const int = comptime formatInt(&buf, value.*, 10, .lower, options);
            state.append(.{ .literal = int }, options);
        },
        .Int => |Int| {
            // this is a bit annoying, basically for each field, we need different logic for comptime vs runtime value
            // could this be done above ? addTokensForRuntime vs addTokensForComptime
            if (comptime_data) |int_slice| {
                comptime var buf: [64]u8 = undefined;
                const value: *const T = @alignCast(@ptrCast(int_slice));
                const int = comptime formatInt(&buf, value.*, 10, .lower, options);
                state.append(.{ .literal = int }, options);
                return;
            }
            const bytes: SmallSlice = .{ .start = base_offset, .end = base_offset + @sizeOf(T) };
            state.append(
                .{ .integer = .{
                    .bytes = bytes,
                    .signedness = Int.signedness,
                    .base = 10,
                } },
                options,
            );
        },
        .Void => {
            if (actual_fmt.len != 0) std.fmt.invalidFmtError(fmt, T);
            state.append(.{ .literal = "void" }, options);
        },
        .Bool => {
            if (actual_fmt.len != 0) std.fmt.invalidFmtError(fmt, T);
            if (comptime_data) |bool_slice| {
                state.append(.{ .boolean = .{ .value = bool_slice[0] > 0, .offset = 0 } }, options);
            } else state.append(.{ .boolean = .{ .offset = 0 } }, options);
        },
        .Optional => |opt| {
            if (actual_fmt.len == 0 or actual_fmt[0] != '?')
                @compileError("cannot format optional without a specifier (i.e. {?} or {any})");
            const bytes: SmallSlice = .{ .start = base_offset, .end = base_offset + @sizeOf(T) };
            state.append(.{ .optional = .{ .bytes = bytes, .skip = 0, .read = ReadOptional(T).read } }, .{});
            const opt_skip: *u8 = &(state.tokens[state.num_tokens - 1].type.optional.skip);
            const opt_idx = state.num_tokens;
            const remaining_fmt = comptime stripOptionalOrErrorUnionSpec(actual_fmt);
            addTokensForType(opt.child, remaining_fmt, state, comptime_data, options, max_depth, base_offset);
            opt_skip.* = state.num_tokens - opt_idx;
        },
        .ErrorUnion => |err_info| {
            if (actual_fmt.len == 0 or actual_fmt[0] != '!')
                @compileError("cannot format error union without a specifier (i.e. {!} or {any})");
            const bytes: SmallSlice = .{ .start = base_offset, .end = base_offset + @sizeOf(T) };
            state.append(.{ .error_union = .{ .bytes = bytes, .skip = 0, .read = ReadErrorUnion(err_info.payload).read } }, .{});
            const err_skip: *u8 = &(state.tokens[state.num_tokens - 1].type.error_union.skip);
            const err_idx = state.num_tokens;
            const remaining_fmt = comptime stripOptionalOrErrorUnionSpec(actual_fmt);
            addTokensForType(err_info.payload, remaining_fmt, state, comptime_data, options, max_depth, base_offset);
            err_skip.* = state.num_tokens - err_idx;
        },
        .Enum, .EnumLiteral, .ErrorSet, .Float, .ComptimeFloat, .Union, .Struct, .Vector => {
            state.append(.not_implemented, .{});
        },
        .Pointer => |ptr_info| switch (ptr_info.size) {
            .One => switch (@typeInfo(ptr_info.child)) {
                .Array, .Enum, .Union, .Struct => {
                    return addTokensForType(ptr_info.child, actual_fmt, state, comptime_data, options, max_depth, base_offset);
                },
                else => state.append(.not_implemented, .{}),
            },
            .Slice => {
                if (actual_fmt.len == 0)
                    @compileError("cannot format slice without a specifier (i.e. {s} or {any})");
                if (max_depth == 0) {
                    return state.append(.{ .literal = "{ ... }" }, .{});
                }
                if (actual_fmt[0] == 's' and ptr_info.child == u8) {
                    return state.append(.buff, options);
                }
                state.append(.not_implemented, .{});
            },
            else => {
                state.append(.not_implemented, .{});
            },
        },
        .Array => |info| {
            if (actual_fmt.len == 0)
                @compileError("cannot format array without a specifier (i.e. {s} or {any})");
            if (max_depth == 0) {
                return state.append(.{ .literal = "{ ... }" }, .{});
            }
            if (actual_fmt[0] == 's' and info.child == u8) {
                if (comptime_data) |comptime_bytes| {
                    return state.append(.{ .literal = comptime_bytes }, options);
                }
                return state.append(.{ .arr = .{ .start = 0, .end = info.len } }, options);
            }
            state.append(.not_implemented, .{});
        },
        .Fn => @compileError("unable to format function body type, use '*const " ++ @typeName(T) ++ "' for a function pointer type"),
        .Type => {
            if (actual_fmt.len != 0) invalidFmtError(fmt, T);
            state.append(.{ .literal = "void" }, options);
        },
        // .EnumLiteral => {
        //     const buffer = [_]u8{'.'} ++ @tagName(value);
        //     return .{ .type = .{ .literal = buffer, } options;
        // },
        .Null => {
            if (actual_fmt.len != 0) invalidFmtError(fmt, T);
            state.append(.{ .literal = "null" }, options);
        },
        else => @compileError("unable to format type '" ++ @typeName(T) ++ "'"),
    }
}

pub fn addAddressTokens(state: *FmtState, comptime T: type) void {
    comptime var type_name: []const u8 = undefined;
    var ptr_offset: u16 = 0;
    var valid: bool = false;
    switch (@typeInfo(T)) {
        .Pointer => |info| {
            valid = true;
            type_name = @typeName(info.child);
            comptime {
                if (info.size == .Slice) {
                    // TODO: ptr_offset = @offsetOf(T, "ptr");
                    ptr_offset = 0;
                }
            }
        },
        .Optional => |info| {
            valid = @typeInfo(info.child) == .Pointer;
            type_name = @typeName(info.child);
        },
        else => {},
    }
    if (!valid)
        @compileError("cannot format non-pointer type " ++ @typeName(T) ++ " with * specifier");

    state.append(.{ .literal = type_name ++ "@" }, .{});
    state.append(.{ .integer = .{
        .bytes = .{ .start = ptr_offset, .end = ptr_offset + @sizeOf(usize) },
        .base = 16,
        .signedness = .unsigned,
    } }, .{});
}

/// This function is intended to be used only in tests. When the formatted result of the template
/// and its arguments does not equal the expected text, it prints diagnostics to stderr to show how
/// they are not equal, then returns an error.
pub fn expectFmt(expected: []const u8, comptime template: []const u8, args: anytype) !void {
    var buff: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buff);
    try format(fbs.writer(), template, args);
    const result = fbs.getWritten();

    if (std.mem.eql(u8, result, expected)) return;
    const stderr = io.getStdErr().writer();
    _ = try stderr.write("\n====== expected this output: =========\n");
    _ = try stderr.write(expected);
    _ = try stderr.write("\n======== instead found this: =========\n");
    _ = try stderr.write(result);
    _ = try stderr.write("\n======================================\n");
    return error.TestExpectedFmt;
}

// test "fmt2 tests" {
// test "fmt2 tests" {
pub fn main() u8 {
    fmt2_test() catch return 1;
    return 0;
}

pub fn fmt2_test() !void {
    var x: i32 = 1;
    x = 2;
    var y: ?i32 = null;

    // Working
    try expectFmt("null: null", "null: {}", .{null});
    try expectFmt("runtime int: 2", "runtime int: {d}", .{x});
    try expectFmt("bool: false", "bool: {}", .{x == 0});
    try expectFmt("comptime bool: true", "comptime bool: {}", .{true});
    try expectFmt("comptime bool: false", "comptime bool: {}", .{false});
    try expectFmt("optional: null", "optional: {?}", .{y});
    y = 5;
    try expectFmt("optional: 5", "optional: {?}", .{y});

    const zig_version = std.SemanticVersion{ .major = 0, .minor = 12, .patch = 0 };
    try expectFmt("0.12.0", "{d}.{d}.{d}", .{ zig_version.major, zig_version.minor, zig_version.patch });
    try expectFmt("0.12.2", "{d}.{d}.{d}", .{ zig_version.major, zig_version.minor, x });
    try expectFmt("0.5.2", "{d}.{d}.{d}", .{ zig_version.major, y.?, x });

    var value: *align(1) i32 = undefined;
    value = @ptrFromInt(0xdeadbeef);
    try expectFmt("pointer: i32@deadbeef\n", "pointer: {*}\n", .{value});

    var hello: []const u8 = "";
    hello = "hello";
    try expectFmt("slice: hello", "slice: {s}", .{hello});
    var opt_hello: ?[]const u8 = null;
    try expectFmt("optional slice: null", "optional slice: {?s}", .{opt_hello});
    opt_hello = hello;
    try expectFmt("optional slice: hello", "optional slice: {?s}", .{opt_hello});
    var hello_arr: [5]u8 = .{ 'h', 'e', 'l', 'l', 'o' };
    hello_arr[0] = 'H';
    try expectFmt("array: Hello", "array: {s}", .{hello_arr});
    try expectFmt("comptime slice: hello", "comptime slice: {s}", .{"hello"});
    try expectFmt("comptime int: 3", "comptime int: {d}", .{3});

    const some_bytes = "\xCA\xFE\xBA\xBE";
    try expectFmt("lowercase: cafebabe\n", "lowercase: {x}\n", .{std.fmt.fmtSliceHexLower(some_bytes)});

    var z: anyerror!i32 = x;
    z = 42;
    try expectFmt("error: 42\n", "error: {!}\n", .{z});
    z = error.OutOfMemory;
    try expectFmt("error: OutOfMemory\n", "error: {!}\n", .{z});

    // Not working
    // try expectFmt("type: u8", "type: {}", .{u8});
    // std.log.warn("{}", .{@typeInfo(@typeInfo(@TypeOf("hello")).Pointer.child)});
    // try expectFmt("pointer: 5@deadbeef\n", "pointer: {any}\n", .{&x});
}
