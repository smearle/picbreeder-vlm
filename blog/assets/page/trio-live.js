/* Interactive archive-growth triptych — the pure-HTML/canvas replacement for the
 * rendered trio_*.mp4 videos. Each `<figure class="trio-live" data-run="…">` gets
 * three canvases (phylogeny growth · Most-Branched leaderboard · archive reel),
 * all drawn from one sprite atlas + JSON (built by tools/build_trio_demo_data.py)
 * and scrubbed from a single clock. Responsive to the figure width; only animates
 * while on screen. Self-contained; multiple instances per page are fine.
 */
(function(){
  "use strict";
  var DPR = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  var LB_RATIO = 0.62, RE_RATIO = 0.50;   // panel width : height (phylo's comes from fw/fh)

  function initTrio(fig){
    var RUN  = fig.getAttribute("data-run");
    var SECS = +(fig.getAttribute("data-secs") || 110);
    var DATA = "assets/trio_demo/" + RUN;
    if (!RUN) return;

    // ---- build DOM (stage + controls) before the figcaption ----
    var stage = document.createElement("div"); stage.className = "trio-live-stage";
    var loading = document.createElement("div"); loading.className = "trio-live-loading";
    loading.textContent = "Loading the archive…"; stage.appendChild(loading);
    var phyCv = mk("canvas"), lbCv = mk("canvas"), reCv = mk("canvas");

    var ctl = document.createElement("div"); ctl.className = "trio-ctl";
    var playBtn = mk("button"); playBtn.type = "button"; playBtn.className = "trio-play";
    playBtn.innerHTML = "&#10073;&#10073;"; playBtn.setAttribute("aria-label", "Pause");
    var seek = mk("input"); seek.type = "range"; seek.className = "trio-seek";
    seek.min = 0; seek.max = 1000; seek.value = 0; seek.step = 1;
    seek.setAttribute("aria-label", "Scrub all three panels in unison");
    var timeEl = mk("span"); timeEl.className = "trio-time"; timeEl.textContent = "…";
    ctl.appendChild(playBtn); ctl.appendChild(seek); ctl.appendChild(timeEl);

    var cap = fig.querySelector("figcaption");
    fig.insertBefore(stage, cap || null);
    fig.insertBefore(ctl, cap || null);

    function mk(t){ return document.createElement(t); }
    function panel(cv, title){
      var d=mk("div"); d.className="trio-panel";
      var h=mk("div"); h.className="trio-panel-title"; h.textContent=title;
      d.appendChild(h); d.appendChild(cv); return d;
    }

    // ---- shared state ----
    var PHY=null, LB=null, atlas=null, N=0, COLS=0, CELL=0, lbPanel=null;
    var childrenIdx=[], px=[], py=[], ready=false;
    var PH_W=0, PH_H=0, LB_W=0, LB_H=0, RE_W=0, RE_H=0;
    var phyCtx, lbCtx, reCtx;

    function cellUV(i){ return [ (i % COLS) * CELL, ((i / COLS) | 0) * CELL ]; }

    function sizeCanvas(cv, w, h){
      cv.style.width=w+"px"; cv.style.height=h+"px";
      cv.width=Math.round(w*DPR); cv.height=Math.round(h*DPR);
      var ctx=cv.getContext("2d"); ctx.setTransform(DPR,0,0,DPR,0,0);
      return ctx;
    }

    function relayout(){
      var gap=14;
      var Wc = stage.clientWidth || fig.clientWidth || 960;
      var ratio = (PHY.fw/PHY.fh) + LB_RATIO + RE_RATIO;
      var H = Math.max(340, Math.min(720, (Wc - 2*gap) / ratio));
      PH_H=LB_H=RE_H=Math.round(H);
      PH_W=Math.round(H*PHY.fw/PHY.fh); LB_W=Math.round(H*LB_RATIO); RE_W=Math.round(H*RE_RATIO);
      phyCtx=sizeCanvas(phyCv, PH_W, PH_H);
      lbCtx =sizeCanvas(lbCv,  LB_W, LB_H);
      reCtx =sizeCanvas(reCv,  RE_W, RE_H);
      var m=14, iw=PH_W-2*m, ih=PH_H-2*m;
      px=new Array(N); py=new Array(N);
      for(var i=0;i<N;i++){ px[i]=m+PHY.nodes[i].x*iw; py[i]=m+PHY.nodes[i].y*ih; }
      initPhBuf();
    }

    // ---- helpers ----
    function boardAt(count){
      var tl=LB.timeline, lo=0, hi=tl.length-1, res=null;
      while(lo<=hi){ var mid=(lo+hi)>>1; if(tl[mid].count<=count){ res=tl[mid]; lo=mid+1; } else hi=mid-1; }
      return res ? res.board : [];
    }
    function curChildCount(idx, fInt){
      var ch=childrenIdx[idx], c=0;
      for(var k=0;k<ch.length;k++) if(ch[k]<fInt) c++;
      return c;
    }

    // ---- PHYLOGENY (incremental edge + node buffers; opaque lineage edges) ----
    var edgeBuf=document.createElement("canvas"), nodeBuf=document.createElement("canvas");
    var edgeCtx=null, nodeCtx=null, builtUpTo=0;
    var PH_TH=7, EDGE_W=1.3, EDGE_A=1;

    function initPhBuf(){
      [edgeBuf,nodeBuf].forEach(function(c){ c.width=Math.round(PH_W*DPR); c.height=Math.round(PH_H*DPR); });
      edgeCtx=edgeBuf.getContext("2d"); nodeCtx=nodeBuf.getContext("2d");
      edgeCtx.setTransform(DPR,0,0,DPR,0,0); nodeCtx.setTransform(DPR,0,0,DPR,0,0);
      edgeCtx.lineWidth=EDGE_W; edgeCtx.lineCap="round";
      builtUpTo=0;
    }
    function bakeTo(toInt){
      if(toInt<builtUpTo){ edgeCtx.clearRect(0,0,PH_W,PH_H); nodeCtx.clearRect(0,0,PH_W,PH_H); builtUpTo=0; }
      for(var i=builtUpTo;i<toInt;i++){
        var nd=PHY.nodes[i], p=nd.p, col=nd.col;
        if(p>=0){
          edgeCtx.strokeStyle="rgba("+col[0]+","+col[1]+","+col[2]+","+EDGE_A+")";
          edgeCtx.beginPath(); edgeCtx.moveTo(px[p],py[p]); edgeCtx.lineTo(px[i],py[i]); edgeCtx.stroke();
        }
        var uv=cellUV(i);
        nodeCtx.drawImage(atlas, uv[0],uv[1],CELL,CELL, px[i]-PH_TH/2, py[i]-PH_TH/2, PH_TH, PH_TH);
      }
      builtUpTo=toInt;
    }
    function drawPhylo(f){
      var fInt=Math.min(N, Math.floor(f));
      bakeTo(fInt);
      var ctx=phyCtx;
      ctx.clearRect(0,0,PH_W,PH_H);
      ctx.drawImage(edgeBuf, 0,0, PH_W,PH_H);
      ctx.drawImage(nodeBuf, 0,0, PH_W,PH_H);
      var board=boardAt(fInt);
      for(var b=0;b<board.length;b++){
        var idx=board[b];
        if(idx>=fInt) continue;
        var cnt=curChildCount(idx,fInt);
        var sz=PH_TH*(2.2 + Math.min(1,cnt/40)*1.6);
        var uv2=cellUV(idx);
        ctx.drawImage(atlas, uv2[0],uv2[1],CELL,CELL, px[idx]-sz/2, py[idx]-sz/2, sz, sz);
        var c2=PHY.nodes[idx].col;
        ctx.strokeStyle="rgb("+c2[0]+","+c2[1]+","+c2[2]+")"; ctx.lineWidth=1.4;
        ctx.strokeRect(px[idx]-sz/2, py[idx]-sz/2, sz, sz);
      }
      // spawn flash: fresh edge drawn in FRONT (thicker), fading as it recedes behind
      var SPAWN=7;
      for(var s=fInt-1; s>=Math.max(0,fInt-SPAWN); s--){
        var a=1-(f-(s+1))/SPAWN; if(a<=0) continue;
        var nd=PHY.nodes[s], pp=nd.p, c=nd.col;
        if(pp<0) continue;
        ctx.globalAlpha=a;
        ctx.strokeStyle="rgb("+c[0]+","+c[1]+","+c[2]+")"; ctx.lineWidth=EDGE_W+0.8;
        ctx.beginPath(); ctx.moveTo(px[pp],py[pp]); ctx.lineTo(px[s],py[s]); ctx.stroke();
      }
      ctx.globalAlpha=1;
    }

    // ---- LEADERBOARD ----
    var lbDisp={};
    var GOLD="#DAA520", INK="#1c1c1c", MUT="#8c8c8c", TRACK="#e4e4e4", ROWALT="#f6f6f6";
    var BAR_H=26, HEAD_H=BAR_H+18;
    function ease(cur,target,k){ return cur+(target-cur)*k; }

    function drawLB(f, dt){
      var fInt=Math.min(N, Math.floor(f));
      var ctx=lbCtx;
      ctx.clearRect(0,0,LB_W,LB_H);
      var margin=12, K=LB.K;
      var rowH=(LB_H-HEAD_H-margin)/K;
      var cell=Math.min(rowH-8, 62);

      var bx0=margin, bx1=LB_W-margin, by0=8, bw=bx1-bx0, card=BAR_H;
      ctx.fillStyle=TRACK; roundRect(ctx,bx0,by0,bw,card,4); ctx.fill();
      if(fInt>0){
        var span=Math.max(1,bw-card), last=fInt;
        var xl=bx0+span*((last-1)/Math.max(1,N-1));
        for(var xcol=bx0; xcol<=xl+0.001; xcol+=1){
          var i=Math.round(((xcol-bx0)/span)*(N-1));
          if(i<0) i=0; if(i>=last) continue;
          var uv=cellUV(i);
          ctx.drawImage(atlas, uv[0],uv[1],CELL,CELL, xcol, by0, card, card);
        }
        var uvl=cellUV(last-1);
        ctx.drawImage(atlas, uvl[0],uvl[1],CELL,CELL, xl, by0, card, card);
        ctx.strokeStyle=GOLD; ctx.lineWidth=2; ctx.strokeRect(xl, by0, card, card);
      }

      var board=boardAt(fInt);
      var rankOf={}; board.forEach(function(idx,r){ rankOf[idx]=r; });
      var k = dt ? Math.min(1, dt*9) : 1;
      board.forEach(function(idx,r){
        var ty=HEAD_H + r*rowH;
        if(!lbDisp[idx]) lbDisp[idx]={y:ty, a:0};
        lbDisp[idx].y=ease(lbDisp[idx].y, ty, k);
        lbDisp[idx].a=ease(lbDisp[idx].a, 1, k);
      });
      for(var id in lbDisp){
        if(!(id in rankOf)){ lbDisp[id].a=ease(lbDisp[id].a, 0, k); if(lbDisp[id].a<0.02) delete lbDisp[id]; }
      }
      var ref=board.length ? Math.max.apply(null, board.map(function(idx){return curChildCount(idx,fInt);})) : 1;
      ref=Math.max(1,ref);

      var ids=Object.keys(lbDisp).sort(function(a,b){ return lbDisp[b].a-lbDisp[a].a; });
      ids.forEach(function(idStr){
        var idx=+idStr, st=lbDisp[idStr];
        if(st.a<=0.02) return;
        var rank = (idStr in rankOf) ? rankOf[idStr] : 99;
        drawLBRow(ctx, idx, st.y, st.a, rank, curChildCount(idx,fInt), ref, rowH, cell, margin);
      });
    }
    function drawLBRow(ctx, idx, y, a, rank, cnt, ref, rowH, cell, margin){
      ctx.save(); ctx.globalAlpha=a;
      var band = rank===0 ? "#fff8e8" : (rank%2 ? ROWALT : "#fff");
      ctx.fillStyle=band; ctx.fillRect(margin, y, LB_W-2*margin, rowH-3);
      ctx.fillStyle = rank===0 ? GOLD : MUT;
      ctx.font="bold 20px Georgia,serif"; ctx.textBaseline="middle"; ctx.textAlign="center";
      ctx.fillText(String(rank+1<100?rank+1:""), margin+16, y+rowH/2);
      var tx=margin+34, ty=y+(rowH-cell)/2, uv=cellUV(idx);
      ctx.drawImage(atlas, uv[0],uv[1],CELL,CELL, tx,ty,cell,cell);
      ctx.strokeStyle = rank===0 ? GOLD : "#ddd"; ctx.lineWidth = rank===0?2:1;
      ctx.strokeRect(tx,ty,cell,cell);
      var textX=tx+cell+12;
      ctx.textAlign="left"; ctx.fillStyle=INK; ctx.font="bold 14px Georgia,serif";
      ctx.fillText(trunc(ctx, LB.items[idx].t, LB_W-margin-textX-4), textX, ty+11);
      var bw=Math.min(110, LB_W-margin-textX-46), bh=15, sy=ty+30;
      ctx.fillStyle=TRACK; roundRect(ctx,textX,sy,bw,bh,bh/2); ctx.fill();
      var fw=Math.max(0,Math.min(1,cnt/ref))*bw;
      if(fw>=bh){ ctx.fillStyle=GOLD; roundRect(ctx,textX,sy,fw,bh,bh/2); ctx.fill(); }
      ctx.fillStyle=INK; ctx.font="bold 14px Georgia,serif";
      ctx.fillText(String(cnt), textX+bw+8, sy+bh/2+1);
      ctx.restore();
    }
    function trunc(ctx,t,maxw){
      if(ctx.measureText(t).width<=maxw) return t;
      while(t.length && ctx.measureText(t+"…").width>maxw) t=t.slice(0,-1);
      return t.replace(/\s+$/,"")+"…";
    }
    function roundRect(ctx,x,y,w,h,r){
      r=Math.min(r,h/2,w/2); ctx.beginPath();
      ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r);
      ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath();
    }

    // ---- REEL ----
    function drawReel(f){
      var ctx=reCtx; ctx.clearRect(0,0,RE_W,RE_H);
      var cols=7, gap=4, tile=(RE_W-(cols+1)*gap)/cols, step=tile+gap;
      var vrows=RE_H/step, revealedRows=f/cols, maxRow=Math.floor(revealedRows);
      var fInt=Math.floor(f);
      var first=Math.floor(revealedRows-vrows-1), lastRow=Math.ceil(revealedRows+1);
      if(first<0) first=0;
      for(var r=first;r<lastRow;r++){
        if(r>maxRow) continue;
        var yy=(vrows-revealedRows+r)*step+gap;
        if(yy>RE_H||yy<-step) continue;
        for(var c=0;c<cols;c++){
          var i=r*cols+c; if(i>=N) break;
          if(i>=fInt) continue;   // discrete: each tile pops in at full opacity once published (no fade)
          var uv=cellUV(i), x=gap+c*step;
          ctx.drawImage(atlas, uv[0],uv[1],CELL,CELL, x, Math.round(yy), tile, tile);
        }
      }
    }

    // ---- clock ----
    var playing=true, p=0, last=null, inView=true;
    function renderAt(f, dt){ drawPhylo(f); drawLB(f, dt); drawReel(f); }
    function setProgress(np, dt){
      p=Math.max(0,Math.min(1,np));
      var f=p*N;
      renderAt(f, dt);
      seek.value=Math.round(p*1000);
      timeEl.textContent=Math.min(N,Math.floor(f))+" / "+N;
    }
    function frame(ts){
      if(ready && inView){
        var dt=0;
        if(playing){
          if(last==null) last=ts;
          dt=(ts-last)/1000; last=ts;
          var np=p+dt/SECS; if(np>=1) np=0;
          setProgress(np, dt);
        } else { last=null; setProgress(p, 1/60); }
      } else last=null;
      requestAnimationFrame(frame);
    }
    function setPlaying(v){
      playing=v;
      playBtn.innerHTML=playing?"&#10073;&#10073;":"&#9654;";
      playBtn.setAttribute("aria-label", playing?"Pause":"Play");
    }
    playBtn.addEventListener("click", function(){ setPlaying(!playing); });
    seek.addEventListener("input", function(){
      setPlaying(false); setProgress(seek.value/1000, 1);
    });

    // ---- double-click the leaderboard -> open the full archive viewer ----
    // Navigate to THIS run's live Growth triptych (archive.html), seeked to the frame +
    // play state we're on. The viewer's back arrow returns to this figure.
    lbCv.title = "Double-click to open this run in the archive viewer, at this moment";
    function openViewer(){
      var api = window.PBVLMArchive;
      if(!ready || !api || typeof api.openGrowth !== "function") return false;
      api.openGrowth({
        run: RUN,
        arc:   fig.getAttribute("data-arc")   || undefined,
        seed:  fig.getAttribute("data-seed")  || undefined,
        model: fig.getAttribute("data-model") || undefined,
        progress: p, playing: playing
      });
      return true;
    }
    lbCv.addEventListener("dblclick", function(e){ e.preventDefault(); openViewer(); });

    // The caption's "archive viewer" link carries a static href to the same run; hijack it
    // so it lands on the frame we're on. If the animation hasn't loaded, the href stands.
    var capLink = fig.querySelector(".trio-archive-note a");
    if(capLink) capLink.addEventListener("click", function(e){
      if(openViewer()) e.preventDefault();
    });

    // ---- keyboard: focus the figure, then space=play/pause, arrows scrub a frame ----
    fig.tabIndex = 0;
    fig.addEventListener("keydown", function(e){
      if(!ready) return;
      var tag=(e.target && e.target.tagName)||"";
      if(tag==="INPUT"||tag==="SELECT"||tag==="TEXTAREA") return;  // let the slider use its own arrows
      if(e.key===" "||e.key==="Spacebar"||e.code==="Space"){
        if(tag==="BUTTON") return;                                // native click already toggles
        e.preventDefault(); setPlaying(!playing);
      } else if(e.key==="ArrowLeft"){
        e.preventDefault(); setPlaying(false); setProgress(p-1/N, 1);
      } else if(e.key==="ArrowRight"){
        e.preventDefault(); setPlaying(false); setProgress(p+1/N, 1);
      }
    });

    // pause the clock while the figure is off screen (save CPU; resume on return)
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function(es){ inView = es[0].isIntersecting; })
        .observe(stage);
    }
    var rT=null;
    window.addEventListener("resize", function(){ if(!ready) return; clearTimeout(rT); rT=setTimeout(function(){ relayout(); setProgress(p, 1); }, 150); });

    // ---- boot ----
    Promise.all([
      fetch(DATA+"/phylo.json").then(function(r){return r.json();}),
      fetch(DATA+"/leaderboard.json").then(function(r){return r.json();}),
      new Promise(function(res,rej){ var im=new Image(); im.onload=function(){res(im);}; im.onerror=rej; im.src=DATA+"/atlas.jpg"; })
    ]).then(function(out){
      PHY=out[0]; LB=out[1]; atlas=out[2];
      N=PHY.n; COLS=PHY.cols; CELL=PHY.cell;
      childrenIdx=new Array(N); for(var i=0;i<N;i++) childrenIdx[i]=[];
      for(var j=0;j<N;j++){ var pj=PHY.nodes[j].p; if(pj>=0) childrenIdx[pj].push(j); }
      stage.removeChild(loading);
      stage.appendChild(panel(phyCv, "Phylogeny growth"));
      lbPanel = panel(lbCv, "Most-Branched leaderboard"); lbPanel.className += " trio-lb";
      stage.appendChild(lbPanel);
      stage.appendChild(panel(reCv,  "Publication reel"));
      relayout();
      ready=true;
      setProgress(0, 1);
      requestAnimationFrame(frame);
    }).catch(function(e){
      loading.textContent = "Could not load this figure’s data (" + e.message + ").";
    });
  }

  function boot(){ document.querySelectorAll("figure.trio-live").forEach(initTrio); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
