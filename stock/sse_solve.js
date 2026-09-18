const fs=require('fs');
const html=fs.readFileSync(process.argv[2],'utf8');
const m=html.match(/<script>([\s\S]*?)<\/script>/);
if(!m){console.error('no script');process.exit(1);}
const stub=`
var window=globalThis;
var document={_c:'',set cookie(v){this._c=v;},get cookie(){return this._c;}};
var location={host:"www.sse.com.cn",reload:function(){}};document.location=location;
`;
const post=`\nconsole.log('COOKIE::'+document.cookie);\n`;
const vm=require('vm');
const ctx={console,require};
vm.createContext(ctx);
vm.runInContext(stub+m[1]+post,ctx);
