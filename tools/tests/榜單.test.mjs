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
const { 讀資料網址, 排出榜單, 場景們, 走步道, 排開 } = 環境;

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
  assert.deepEqual(方向("測試成本"), ["越小越好", false]);
  assert.deepEqual(方向("Test Cost"), ["越小越好", false]);
  assert.deepEqual(方向("測試正確率"), ["越大越好", false]);
  assert.deepEqual(方向("Accuracy"), ["越大越好", false]);
  assert.deepEqual(方向("成績"), ["越小越好", true]);
  // 有指定就照指定
  assert.deepEqual([排出榜單(表格(["組別", "Loss"], [[1, 1]]), "越大越好").方向], ["越大越好"]);
});

test("成績欄後面多出的空白欄不算，最後一個有東西的欄才是成績", () => {
  const 榜 = 排出榜單(表格(
    ["組別", "組員", "測試成本", "", ""],
    [[1, "王小明", 0.3, null, null], [2, "李小華", 0.2, null, null]],
  ));
  assert.equal(榜.標題, "測試成本");
  assert.equal(榜.方向, "越小越好");
  assert.deepEqual(本地(榜.組.map((g) => [g.組別, g.成績, g.組員])), [["2", 0.2, ["李小華"]], ["1", 0.3, ["王小明"]]]);
});

test("Google 沒認出表頭時，第一列當表頭，不會變成一組", () => {
  // 成績欄設成純文字時，整張表都是文字，Google 常把表頭當成資料
  const 榜 = 排出榜單(表格(
    ["", "", "", ""],
    [["組別", "組員", "組員", "測試成本"], ["1", "王小明", "李小華", "0.3"], ["2", "陳大同", "", "0.2"]],
  ));
  assert.equal(榜.標題, "測試成本");
  assert.deepEqual(本地(榜.組.map((g) => [g.組別, g.成績, g.組員])),
    [["2", 0.2, ["陳大同"]], ["1", 0.3, ["王小明", "李小華"]]]);
});

test("成績欄右邊還有別的欄，看欄名找成績欄，不把它當組員", () => {
  const 榜 = 排出榜單(表格(
    ["組別", "組員", "測試成本", "備註"],
    [[1, "王小明", 0.3, "第二次交"], [2, "李小華", 0.2, ""]],
  ));
  assert.equal(榜.標題, "測試成本");
  assert.deepEqual(本地(榜.組.map((g) => [g.組別, g.成績, g.組員])),
    [["2", 0.2, ["李小華"]], ["1", 0.3, ["王小明"]]]);
});

test("空的成績表不會出錯，只是沒有組", () => {
  const 榜 = 排出榜單({ cols: [], rows: [] });
  assert.equal(榜.組.length, 0);
});

// ---------- 場景 ----------
const 場景名 = ["山", "大樓", "深海", "火箭", "棒球場", "攀岩牆", "巨樹", "火山", "金字塔"];

test("九個場景都有，步道上每一點都站在場景的可站範圍內", () => {
  assert.deepEqual(本地(Object.keys(場景們)), 場景名);
  for (const 名 of 場景名) {
    const 場景 = 場景們[名];
    const 長 = 走步道(場景, Infinity).s;
    for (let s = 0; s <= 長; s += 5) {
      const { x, y } = 走步道(場景, s);
      const [左, 右] = 場景.範圍(y);
      assert.ok(x >= 左 && x <= 右, `${名}：步道 s=${s} 的 (${x.toFixed(0)}, ${y.toFixed(0)}) 跑到範圍 [${左.toFixed(0)}, ${右.toFixed(0)}] 外面`);
      assert.ok(y >= 0 && y <= 860, `${名}：步道跑出畫面`);
    }
  }
});

test("深海越好潛得越深，棒球場跑一圈回本壘，其他越好越高", () => {
  for (const 名 of 場景名) {
    const 起 = 走步道(場景們[名], 0), 終 = 走步道(場景們[名], Infinity);
    if (名 === "深海") assert.ok(終.y - 起.y > 400, "深海的終點要在海底");
    else if (名 === "棒球場") assert.ok(Math.abs(終.y - 起.y) < 50, "棒球場的終點是本壘，和起點差不多高");
    else assert.ok(起.y - 終.y > 400, `${名}的終點要在高處`);
  }
});

// 一組在畫面上佔的框：旗子頂在腳上面 88，第二行名字底在腳下面 40
const 框 = ({ x, y, 縮放 }, 寬) => ({ 左: x - (寬 * 縮放) / 2, 右: x + (寬 * 縮放) / 2, 上: y - 88 * 縮放, 下: y + 40 * 縮放 });
const 疊到 = (a, b) => a.左 < b.右 && b.左 < a.右 && a.上 < b.下 && b.上 < a.下;
function 檢查排開(名, 上場) {
  const 結果 = 本地(排開(上場, 場景們[名]));
  assert.equal(結果.length, 上場.length);
  const 框們 = 結果.map((r, i) => 框(r, 上場[i].寬));
  框們.forEach((b, i) => {
    const [左, 右] = 場景們[名].範圍(結果[i].y);
    assert.ok(b.左 >= 左 - 0.5 && b.右 <= 右 + 0.5, `${名}：第 ${i} 組排到範圍外面 ${JSON.stringify(b)} 範圍 [${左}, ${右}]`);
    for (let j = 0; j < i; j++) assert.ok(!疊到(b, 框們[j]), `${名}：第 ${i} 組和第 ${j} 組疊在一起`);
  });
  return 結果;
}

test("同分的組擠在終點時並排或往回排，不會疊在一起，也不會排到場景外面", () => {
  for (const 名 of 場景名) {
    const 長 = 走步道(場景們[名], Infinity).s;
    檢查排開(名, Array.from({ length: 8 }, (_, i) => ({ 寬: 56 + (i % 3) * 40, s: 長 })));
  }
});

test("同分的幾組在窄的山頂放不下時，整排一起往下退，還是並排站在同一排", () => {
  for (const 名 of ["山", "火山", "金字塔"]) {
    const 長 = 走步道(場景們[名], Infinity).s;
    const 結果 = 檢查排開(名, [{ 寬: 96, s: 長 }, { 寬: 96, s: 長 }, { 寬: 56, s: 長 }]);
    assert.ok(結果.every((r) => Math.abs(r.y - 結果[0].y) < 1e-6), `${名}：同分的組沒有站在同一排 ${JSON.stringify(結果.map((r) => Math.round(r.y)))}`);
  }
});

test("分數不同的組擠在起點附近，退到起點也不會疊在一起", () => {
  檢查排開("火箭", [{ 寬: 176, s: 5 }, { 寬: 176, s: 129 }, { 寬: 176, s: 853 }, { 寬: 96, s: 152 }]);
  for (const 名 of 場景名) 檢查排開(名, [0, 10, 30, 60].map((s) => ({ 寬: 136, s })));
});

test("棒球場一壘和三壘的組各自待在自己的壘包，不會被併成一排", () => {
  const 球場 = 場景們.棒球場;
  // 步道：準備區 → 本壘 → 一壘 → 二壘 → 三壘 → 本壘，找出一壘、三壘在步道上的 s
  const 找 = (x, y) => { for (let s = 0; ; s += 1) { const p = 走步道(球場, s); if (Math.hypot(p.x - x, p.y - y) < 1) return s; } };
  const 一壘 = 找(960, 480), 三壘 = 找(240, 480);
  const [甲, 乙] = 檢查排開("棒球場", [{ 寬: 96, s: 一壘 }, { 寬: 96, s: 三壘 }]);
  assert.ok(Math.hypot(甲.x - 960, 甲.y - 480) < 60, `一壘的組跑到 (${甲.x}, ${甲.y})`);
  assert.ok(Math.hypot(乙.x - 240, 乙.y - 480) < 60, `三壘的組跑到 (${乙.x}, ${乙.y})`);
});

test("被擠開的組只往起點退，不會跑到分數比它好的組前面", () => {
  let 種子 = 7;
  const 亂 = () => ((種子 = (種子 * 16807) % 2147483647) / 2147483647);
  for (const 名 of 場景名) {
    const 長 = 走步道(場景們[名], Infinity).s;
    for (let 回 = 0; 回 < 20; 回++) {
      // 分數擠在幾個值附近，製造很多擠在一起的情況
      const 上場 = Array.from({ length: 4 + Math.floor(亂() * 9) }, () => ({ 寬: 56 + Math.floor(亂() * 4) * 40, s: Math.round(亂() * 4) / 4 * 長 }));
      const 結果 = 排開(上場, 場景們[名]);
      上場.forEach((u, i) => {
        assert.ok(結果[i].s <= u.s + 1e-9, `${名}：組往前跑了`);
        上場.forEach((v, j) => { if (u.s > v.s) assert.ok(結果[i].s >= 結果[j].s - 1e-9, `${名}：分數低的組排到分數高的前面`); });
      });
    }
  }
});

test("最後一名和第一名不會擠在同一個地方（四人組也放得下）", () => {
  for (const 名 of 場景名) {
    const 起 = 走步道(場景們[名], 0), 終 = 走步道(場景們[名], Infinity);
    assert.ok(Math.hypot(終.x - 起.x, 終.y - 起.y) >= 200, `${名}的起點和終點太近`);
  }
});
