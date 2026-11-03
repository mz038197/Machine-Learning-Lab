// 榜單.html 的邏輯測試。執行：node --test "tools/tests/*.test.mjs"
// 從 榜單.html 抓出 <script id="榜單邏輯"> 那一段，在乾淨的環境裡跑。
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const html = readFileSync(new URL("../榜單.html", import.meta.url), "utf8");
const 邏輯 = html.match(/<script id="榜單邏輯">([\s\S]*?)<\/script>/)[1];
const 環境 = vm.createContext({});
vm.runInContext(邏輯, 環境);
const { 讀資料網址, 排出榜單 } = 環境;

// 照 Google 回傳的樣子造一張表：數字格有 v 和 f，文字格只有 v，空格是 null。
function 表格(欄名, 列) {
  return {
    cols: 欄名.map((label) => ({ label })),
    rows: 列.map((r) => ({
      c: r.map((x) =>
        x === null || x === "" ? null : typeof x === "number" ? { v: x, f: String(x) } : { v: x }),
    })),
  };
}

// 只留測試在意的欄位，方便比對。
// vm 裡造出來的陣列跟這裡的 Array 不是同一個，deepEqual 前先轉成這邊的物件。
const 本地 = (x) => JSON.parse(JSON.stringify(x));
const 簡要 = (榜) => 本地(榜.組.map(({ 組別, 名次, 高度 }) => ({ 組別, 名次, 高度 })));

const ID = "1JqCphSygWU0_M1JH6tFa_z7Sn2T6prKJW2IDcsQu-qA";

test("分享連結轉成讀資料網址", () => {
  assert.equal(
    讀資料網址(`https://docs.google.com/spreadsheets/d/${ID}/edit?usp=drive_link`, "cb"),
    `https://docs.google.com/spreadsheets/d/${ID}/gviz/tq?tqx=out:json;responseHandler:cb&headers=1`,
  );
});

test("連結指定了工作表，就讀那一張", () => {
  assert.equal(
    讀資料網址(`https://docs.google.com/spreadsheets/d/${ID}/edit#gid=123456`, "cb"),
    `https://docs.google.com/spreadsheets/d/${ID}/gviz/tq?tqx=out:json;responseHandler:cb&headers=1&gid=123456`,
  );
  assert.equal(
    讀資料網址(`https://docs.google.com/spreadsheets/d/${ID}/edit?gid=78#gid=78`, "cb"),
    `https://docs.google.com/spreadsheets/d/${ID}/gviz/tq?tqx=out:json;responseHandler:cb&headers=1&gid=78`,
  );
});

test("不是 Google 試算表的連結，說清楚哪裡不對", () => {
  assert.throws(() => 讀資料網址("https://drive.google.com/file/d/abc/view", "cb"), /不是 Google 試算表的連結/);
  assert.throws(() => 讀資料網址("", "cb"), /不是 Google 試算表的連結/);
});

test("越小越好：最小的在山頂，最大的在山腳，中間照比例", () => {
  const 榜 = 排出榜單(
    表格(["組別", "組員1", "組員2", "Loss"], [
      [1, "王小明", "李小華", 30],
      [2, "陳大文", "林小美", 10],
      [3, "張三", "李四", 20],
    ]),
    "越小越好",
  );
  assert.deepEqual(簡要(榜), [
    { 組別: "2", 名次: 1, 高度: 1 },
    { 組別: "3", 名次: 2, 高度: 0.5 },
    { 組別: "1", 名次: 3, 高度: 0 },
  ]);
});

test("越大越好：最大的在山頂", () => {
  const 榜 = 排出榜單(
    表格(["組別", "組員1", "組員2", "測試正確率"], [
      [1, "甲", "乙", 0.6],
      [2, "丙", "丁", 0.9],
      [3, "戊", "己", 0.7],
    ]),
    "越大越好",
  );
  assert.deepEqual(簡要(榜).map((g) => [g.組別, g.名次]), [["2", 1], ["3", 2], ["1", 3]]);
  assert.equal(榜.組[0].高度, 1);
  assert.equal(榜.組[2].高度, 0);
  assert.ok(Math.abs(榜.組[1].高度 - 1 / 3) < 1e-9);
});

test("沒填和讀不懂的組留在營地，不算名次，排在最後", () => {
  const 榜 = 排出榜單(
    表格(["組別", "組員1", "組員2", "Loss"], [
      [1, "甲", "乙", ""],
      [2, "丙", "丁", "還在跑"],
      [3, "戊", "己", 50],
      [4, "庚", "辛", 40],
    ]),
    "越小越好",
  );
  assert.deepEqual(
    榜.組.map(({ 組別, 狀態, 名次, 高度, 成績原文 }) => [組別, 狀態, 名次, 高度, 成績原文]),
    [
      ["4", "已填", 1, 1, "40"],
      ["3", "已填", 2, 0, "50"],
      ["1", "未填", null, null, ""],
      ["2", "讀不懂", null, null, "還在跑"],
    ],
  );
});

test("學生各種寫法的數字都讀得懂", () => {
  const 寫法 = [
    ["1.2e5", 120000],
    ["3,456.7", 3456.7],
    ["１２．５", 12.5],
    ["  88  ", 88],
    ["MSE=12.3", 12.3],
    ["約 300", 300],
    ["85%", 0.85],
    ["-0.5", -0.5],
    [".75", 0.75],
  ];
  const 榜 = 排出榜單(
    表格(["組別", "成績"], 寫法.map(([w], i) => [i + 1, w])),
    "越小越好",
  );
  const 依組別 = Object.fromEntries(榜.組.map((g) => [g.組別, g.成績]));
  寫法.forEach(([w, 數], i) => assert.equal(依組別[String(i + 1)], 數, `「${w}」應該讀成 ${數}`));
});

test("同分同名次，下一名跳號", () => {
  const 榜 = 排出榜單(
    表格(["組別", "Loss"], [[1, 5], [2, 3], [3, 3], [4, 9]]),
    "越小越好",
  );
  assert.deepEqual(簡要(榜).map((g) => [g.組別, g.名次, g.高度]), [
    ["2", 1, 1], ["3", 1, 1], ["1", 3, 4 / 6], ["4", 4, 0],
  ]);
});

test("只有一組有填，或大家一樣，都站在山頂", () => {
  const 一組 = 排出榜單(表格(["組別", "Loss"], [[1, ""], [2, 7]]), "越小越好");
  assert.deepEqual(簡要(一組).map((g) => [g.組別, g.名次, g.高度]), [["2", 1, 1], ["1", null, null]]);
  const 一樣 = 排出榜單(表格(["組別", "Loss"], [[1, 7], [2, 7]]), "越小越好");
  assert.deepEqual(簡要(一樣).map((g) => [g.名次, g.高度]), [[1, 1], [1, 1]]);
});

test("中間的欄都是組員，空的不算；成績欄的欄名就是標題；組別空白的列跳過", () => {
  const 榜 = 排出榜單(
    表格(["組別", "組員1", "組員2", "組員3", "測試 MSE"], [
      [1, "王小明", "", "李小華", 12],
      [null, "", "", "", ""],
      [2, "", "", "", 15],
    ]),
    "越小越好",
  );
  assert.equal(榜.標題, "測試 MSE");
  assert.deepEqual(
    本地(榜.組.map((g) => [g.組別, g.組員])),
    [["1", ["王小明", "李小華"]], ["2", []]],
  );
});

test("沒指定方向時看欄名猜；猜不到就當越小越好，並說是猜不到", () => {
  const 方向 = (欄名) => {
    const 榜 = 排出榜單(表格(["組別", 欄名], [[1, 1]]));
    return [榜.方向, 榜.方向是預設的];
  };
  assert.deepEqual(方向("Loss"), ["越小越好", false]);
  assert.deepEqual(方向("測試 MSE"), ["越小越好", false]);
  assert.deepEqual(方向("誤差"), ["越小越好", false]);
  assert.deepEqual(方向("測試正確率"), ["越大越好", false]);
  assert.deepEqual(方向("Accuracy"), ["越大越好", false]);
  assert.deepEqual(方向("成績"), ["越小越好", true]);
  // 有指定就照指定
  assert.deepEqual([排出榜單(表格(["組別", "Loss"], [[1, 1]]), "越大越好").方向], ["越大越好"]);
});

test("空的成績表不會出錯，只是沒有組", () => {
  const 榜 = 排出榜單({ cols: [], rows: [] });
  assert.equal(榜.組.length, 0);
});
